import time

import numpy as np
import torch
import random

from sklearn.cluster import DBSCAN, KMeans

from config import cfg
import os
import os.path as osp
import pickle
import json
import torch.nn.functional as F
import torch.distributed
from tqdm import tqdm
from collections import Counter
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from collections import defaultdict
import torch.optim as op
import scipy.sparse as sp

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class MultiHeadGraphAttention(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    """

    def __init__(self, n_head, f_in, f_out, attn_dropout, diag=True, init=None, bias=False):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.diag = diag
        if self.diag:
            self.w = Parameter(torch.Tensor(n_head, 1, f_out))
        else:
            self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src_dst = Parameter(torch.Tensor(n_head, f_out * 2, 1))
        self.attn_dropout = attn_dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.special_spmm = SpecialSpmm()
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        if init is not None and diag:
            init(self.w)
            stdv = 1. / math.sqrt(self.a_src_dst.size(1))
            nn.init.uniform_(self.a_src_dst, -stdv, stdv)
        else:
            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src_dst)

    def forward(self, input, adj):
        output = []
        for i in range(self.n_head):
            # pdb.set_trace()
            N = input.size()[0]
            edge = adj._indices()
            if self.diag:
                h = torch.mul(input, self.w[i])
            else:
                h = torch.mm(input, self.w[i])

            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1)  # edge: 2*D x E
            edge_e = torch.exp(-self.leaky_relu(edge_h.mm(self.a_src_dst[i]).squeeze()))  # edge_e: 1 x E
            # e_rowsum: N x 1
            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1)).cuda() if next(self.parameters()).is_cuda else torch.ones(size=(N, 1)))
            # pdb.set_trace()
            edge_e = F.dropout(edge_e, self.attn_dropout, training=self.training)   # edge_e: 1 x E

            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            h_prime = h_prime.div(e_rowsum)

            output.append(h_prime.unsqueeze(0))
        # pdb.set_trace()
        output = torch.cat(output, dim=0)
        # pdb.set_trace()
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        if self.diag:
            return self.__class__.__name__ + ' (' + str(self.f_out) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'
        else:
            return self.__class__.__name__ + ' (' + str(self.f_in) + ' -> ' + str(self.f_out) + ') * ' + str(self.n_head) + ' heads'

class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(GAT, self).__init__()
        self.linear = nn.Linear(2048,300)
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False))

    def forward(self, x, adj):
        x = self.linear(x)
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)

            x = gat_layer(x, adj)

            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x

class UnspervisedModel(nn.Module):
    def __init__(self, rel_features, attr_features, img_features, adj):
        super().__init__()
        self.img = img_features
        self.adj = adj
        self.rel = rel_features
        self.attr = attr_features
        self.stru_gat = GAT(n_units=[300, 300, 300], n_heads=[2, 2], dropout=0.0,
                            attn_dropout=0.0,
                            instance_normalization=False, diag=True)
        # self.stru_gcn = GCN(300,300,300,0.0)
        self.Att_linear = nn.Linear(768, 300)
        # self.Att_mlp = nn.Sequential(nn.Linear(768, 500), nn.ReLU(), nn.Linear(500, 300))
        self.Rel_linear = nn.Linear(768, 300)
        # self.Rel_mlp = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 300))
        self.Img_linear = nn.Linear(2048, 300)

    def forward(self):
        if self.img is not None:
            stru_gat = self.stru_gat(self.img, self.adj)
            img_linear = self.Img_linear(self.img)
        else:
            stru_gat = None
            img_linear = None
        if self.rel is not None:
            rel_linear = self.Rel_linear(self.rel)
        else:
            rel_linear = None
        att_linear = self.Att_linear(self.attr)

        return stru_gat, rel_linear, att_linear, img_linear


def test(left_ents, right_ents, img_features,attr_features,rel_features,ills):

    l_attr_f = attr_features[left_ents]  # left images
    r_attr_f = attr_features[right_ents]
    attr_sim = l_attr_f.mm(r_attr_f.t())
    l_rel_f = rel_features[left_ents]
    r_rel_f = rel_features[right_ents]
    if  img_features is not None:
        l_img_f = img_features[left_ents]  # left images
        r_img_f = img_features[right_ents]  # right images
        all_sim = l_img_f.mm(r_img_f.t()) * 0.8 + attr_sim * 0.1 + 0.1 * l_rel_f.mm(r_rel_f.t())
    else:
        all_sim = attr_sim * 0.1 + 0.1 * l_rel_f.mm(r_rel_f.t())

    # 记录行号
    incorrect_rows = []

    for i in range(all_sim.size(0)):
        if all_sim[i, i] != torch.max(all_sim[i, :]):
            incorrect_rows.append(i)
    return incorrect_rows



def test_judge(visual_links,ills):
    count = 0.0
    before = len(visual_links) / 2
    cou = 0
    count_q = 0
    count_b = 0
    for link in visual_links:
        cou = cou + 1
        if link in ills:
            count = count + 1
        else:
            if cou < before:
                count_q += 1
            else:
                count_b += 1
    print(
        f"{(count / len(ills) * 100):.2f}% in true links，{(count / len(visual_links) * 100):.2f}% in true links")
    print(f"visual links length: {(len(visual_links))}")

def get_topk_indices(M, K=1000):
    H, W = M.shape
    M_view = M.view(-1)
    vals, indices = M_view.topk(K)
    print("highest sim:", vals[0].item(), "lowest sim:", vals[-1].item())
    two_d_indices = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1)
    return two_d_indices,vals[0].item(),vals[-1].item()

def stage_one(args, left_ents, right_ents, img_features,attr_features,rel_features, ills):
    entity_num = args.unsup_k
    n_cal = args.cluster_num
    alpha = 0.8
    beta = 0.1
    gamma = 0.1

    if img_features is not None:
        l_img_f = torch.cat([alpha*img_features[left_ents],beta *F.normalize(attr_features[left_ents]),gamma*F.normalize(rel_features[left_ents])],dim=1)
        r_img_f = torch.cat([alpha*img_features[right_ents],beta *F.normalize(attr_features[right_ents]),gamma*F.normalize(rel_features[right_ents])],dim=1)
    else:
        l_img_f = torch.cat([beta * F.normalize(attr_features[left_ents]),gamma*F.normalize(rel_features[left_ents])], dim=1)
        r_img_f = torch.cat([ beta * F.normalize(attr_features[right_ents]),gamma*F.normalize(rel_features[right_ents])], dim=1)



    all_img_f = np.vstack((l_img_f, r_img_f))
    all_ents = np.concatenate((left_ents, right_ents))
    kmeans = KMeans(n_clusters=n_cal, random_state=0).fit(all_img_f)
    labels = kmeans.labels_
    cluster_indices = {i: {'left': [], 'right': []} for i in range(n_cal)}
    for label, ent in zip(labels, all_ents):
        if ent in left_ents:
            cluster_indices[label]['left'].append(ent)
        else:
            cluster_indices[label]['right'].append(ent)
    left_cluster_features = {i: torch.cat([rel_features[np.array(idxs['left'])],attr_features[np.array(idxs['left'])]],dim=1 ) for i, idxs in cluster_indices.items()}
    right_cluster_features = {i: torch.cat([rel_features[np.array(idxs['right'])],attr_features[np.array(idxs['right'])]],dim=1 ) for i, idxs in cluster_indices.items()}
    similarity_matrices = {}

    for i in range(n_cal):
        if len(left_cluster_features[i]) > 0 and len(right_cluster_features[i]) > 0:
            similarity_matrices[i] = left_cluster_features[i].mm(right_cluster_features[i].t()).cuda()
        else:
            similarity_matrices[i] = None

    visual_links = []
    used_inds = []
    for i in range(n_cal):
        if similarity_matrices[i] is None:
            continue
        else:
            topk = int(entity_num*(sum(similarity_matrices[i].shape)/len(all_ents)))
        two_d_indices,_,_ = get_topk_indices(similarity_matrices[i], topk * 100)
        count = 0
        for ind in two_d_indices:
            if cluster_indices[i]['left'][ind[0]] in used_inds:
                continue
            if cluster_indices[i]['right'][ind[1]] in used_inds:
                continue
            used_inds.append(cluster_indices[i]['left'][ind[0]])
            used_inds.append(cluster_indices[i]['right'][ind[1]])
            visual_links.append((cluster_indices[i]['left'][ind[0]], cluster_indices[i]['right'][ind[1]]))
            count += 1
            if count == topk:
                break

    count = 0.0
    for link in visual_links:
        if link in ills:
            count = count + 1
    print(
        f"{(count / len(ills) * 100):.2f}% in true links，{(count / len(visual_links) * 100):.2f}% in true links")
    print(f"visual links length: {(len(visual_links))}")
    train_ill = np.array(visual_links, dtype=np.int32)
    with open(f'{args.cluster_num}_{args.data_split}_{args.unsup_k}_1.pkl', 'wb') as file:
        pickle.dump(train_ill, file)
    return train_ill
def softXEnt(target, logits, replay=False, neg_cross_kg=False):
    # torch.Size([2239, 4478])

    logprobs = F.log_softmax(logits, dim=1)
    loss = -(target * logprobs).sum() / logits.shape[0]

    return loss
def icl_loss(zis, output_zis, output_zjs):
    n_view = 2
    temperature = 1
    LARGE_NUM = 1e9
    num_classes = len(zis) * n_view
    hidden1, hidden2 = output_zis, output_zjs
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = F.one_hot(torch.arange(start=0, end=len(zis), dtype=torch.int64),
                       num_classes=num_classes).float()
    labels = labels.cuda()

    masks = F.one_hot(torch.arange(start=0, end=len(zis), dtype=torch.int64),
                      num_classes=len(zis))
    masks = masks.cuda().float()
    logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature
    logits_a = torch.cat([logits_ab, logits_aa], dim=1)
    logits_b = torch.cat([logits_ba, logits_bb], dim=1)
    loss_a = softXEnt(labels, logits_a).cuda()
    loss_b = softXEnt(labels, logits_b).cuda()
    loss = 0.5 * loss_a + loss_b * 0.5
    return loss


def test_model(model, left_ents, right_ents, alpha, beta, gamma, train_ill, img_features, attr_features,
               rel_features, ills,epoch,args):
    model.eval()
    with torch.no_grad():
        stru_gat, rel_linear, att_linear, img_linear = model()

    left_ents2 = left_ents  # list(set(left_ents) - set(train_ill[:, 0]))
    right_ents2 = right_ents  # list(set(right_ents)-set(train_ill[:,1]))

    #todo
    l_rel_f = F.normalize(rel_linear[left_ents2])
    r_rel_f = F.normalize(rel_linear[right_ents2])
    l_attr_f = F.normalize(att_linear[left_ents2])
    r_attr_f = F.normalize(att_linear[right_ents2])
    if img_features is not None:
        l_stru_f = F.normalize(stru_gat[left_ents2])
        r_stru_f = F.normalize(stru_gat[right_ents2])
        l_all_f = torch.cat([alpha * l_stru_f, beta * l_attr_f,gamma*l_rel_f], dim=1)
        r_all_f = torch.cat([alpha * r_stru_f, beta * r_attr_f,gamma*r_rel_f], dim=1)
    else:
        l_all_f = torch.cat([beta * l_attr_f,gamma*l_rel_f], dim=1)
        r_all_f = torch.cat([beta * r_attr_f,gamma*r_rel_f], dim=1)

    #####新加
    all_sim = l_all_f.mm(r_all_f.t())
    topk =  args.unsup_k # args.unsup_k
    two_d_indices, _, _ = get_topk_indices(all_sim, topk * 100)

    visual_links = []
    used_inds = []
    count = 0
    used_l_inds = []
    count_a = 0
    for ind in two_d_indices:
        if left_ents[ind[0]] in used_inds:
            continue
        if right_ents[ind[1]] in used_inds:
            continue
        else:
            count_a += 1
            used_inds.append(left_ents[ind[0]])
            used_inds.append(right_ents[ind[1]])
            used_l_inds.append(left_ents[ind[0]])
            visual_links.append((left_ents[ind[0]], right_ents[ind[1]]))
            count += 1
        if count == topk:
            break

    news_ill = np.array(visual_links, dtype=np.int32)
    train_ill_flat = train_ill.flatten()
    for news in news_ill:
        new_data_flat = news.flatten()
        # 检查 new_data 中的元素是否在 trains_ill 中存在
        existing_elements = np.isin(new_data_flat, train_ill_flat)
        if not existing_elements.any():
            # 如果不存在，则添加
            train_ill = np.vstack((train_ill, news))

    incorrect_rows = test(train_ill[:, 0], train_ill[:, 1], img_features, attr_features, rel_features, ills)

    visual_links = [tuple(row) for row in train_ill.tolist()]
    test_judge(visual_links, ills)
    if epoch==299:
        train_ill = np.delete(train_ill, incorrect_rows, axis=0)
        with open(f'{args.cluster_num}_{args.data_split}_{args.unsup_k}_2.pkl', 'wb') as file:
            pickle.dump(train_ill, file)
        print("stage 2 is over!")
        visual_links = [tuple(row) for row in train_ill.tolist()]
        test_judge(visual_links, ills)

def stage_two(args, left_ents, right_ents, img_features,attr_features,rel_features,ills, adj):
    alpha = 0.8
    beta = 0.1
    gamma = 0.1
    if img_features is not None:
        img_features = img_features.cuda()
    attr_features = attr_features.cuda()
    if rel_features is not None:
        rel_features = rel_features.cuda()
    adj =adj.cuda()
    model = UnspervisedModel(rel_features, attr_features, img_features, adj).cuda()
    with open(f'/home/hongyunpeng/unsup_test/{args.cluster_num}_{args.data_split}_{args.unsup_k}_1.pkl', 'rb') as file:
        train_ill = pickle.load(file)

    all_ents = train_ill

    lr = 0.01
    params = model.parameters()
    optimizer = op.AdamW(params, lr=lr)
    batch_zise = 2000
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        stru_gat, rel_linear, att_linear, img_linear = model()
        length = len(all_ents)
        for si in np.arange(0, length, batch_zise):
            zis = all_ents[si:si+batch_zise][:,0]
            zjs = all_ents[si:si+batch_zise][:,1]

            if img_features is not None:
                s_zis_p = stru_gat[zis].cuda()
                s_zjs_p = stru_gat[zjs].cuda()

                loss1 = icl_loss(s_zis_p,s_zis_p,s_zjs_p)

            if rel_features is not None:
                r_zis_p = rel_linear[zis].cuda()
                r_zjs_p = rel_linear[zjs].cuda()

                loss2 =  icl_loss(r_zis_p,r_zis_p,r_zjs_p)

            a_zis_p = att_linear[zis].cuda()
            a_zjs_p = att_linear[zjs].cuda()
            loss3 = icl_loss(a_zis_p, a_zis_p, a_zjs_p)

            if img_features is not None:
                loss = loss1+loss2+loss3
            else:
                loss = loss2+loss3
            loss.backward(retain_graph=True)
            print(f'loss:{loss},epoch:{epoch}')
            del  loss
        optimizer.step()
        test_model(model, left_ents, right_ents, alpha, beta, gamma, train_ill, img_features, attr_features,
                   rel_features, ills,epoch,args)

    torch.save(model.state_dict(), f'model_{args.cluster_num}_{args.data_split}_{args.unsup_k}_1.pth')




    return train_ill


def stage_three(args, img_features,attr_features,rel_features,ills, adj,dic_adj):
    if img_features is not None:
        img_features = img_features.cuda()
    attr_features = attr_features.cuda()
    if rel_features is not None:
        rel_features = rel_features.cuda()
    adj =adj.cuda()
    model = UnspervisedModel(rel_features, attr_features, img_features, adj).cuda()
    with open(f'/home/hongyunpeng/unsup_test/{args.cluster_num}_{args.data_split}_{args.unsup_k}_2.pkl', 'rb') as file:
        train_ill = pickle.load(file)

    left_neibor = [dic_adj[key] if key in dic_adj else {key}  for key in train_ill[:, 0]]
    right_neibor = [dic_adj[key] if key in dic_adj else {key}  for key in train_ill[:, 1]]
    left_neibor_emd_i, left_neibor_emd_a, left_neibor_emd_r = [], [], []
    right_neibor_emd_i, right_neibor_emd_a, right_neibor_emd_r = [], [], []

    model.load_state_dict(torch.load(f'model_{args.cluster_num}_{args.data_split}_{args.unsup_k}_1.pth'))
    model.eval()
    with torch.no_grad():
        stru_gat, rel_linear, att_linear, img_linear = model()

    alpha = 0.8
    beta = 0.1
    gamma = 0.1

    for item in left_neibor:
        indices = torch.tensor(list(item))
        if img_features is not None:
            left_neibor_emd_i.append(F.normalize(torch.cat([img_features[indices], F.normalize(stru_gat[indices])], dim=1)))
        #todo
        left_neibor_emd_a.append(
            F.normalize(torch.cat([attr_features[indices], F.normalize(att_linear[indices])], dim=1)))
        if rel_features is not None:
            left_neibor_emd_r.append(
            F.normalize(torch.cat([rel_features[indices], F.normalize(rel_linear[indices])], dim=1)))

    for item in right_neibor:
        indices = torch.tensor(list(item))
        if img_features is not None:
            right_neibor_emd_i.append(
                F.normalize(torch.cat([img_features[indices], F.normalize(stru_gat[indices])], dim=1)))
        right_neibor_emd_a.append(
            F.normalize(torch.cat([attr_features[indices], F.normalize(att_linear[indices])], dim=1)))
        if rel_features is not None:
            right_neibor_emd_r.append(F.normalize(torch.cat([rel_features[indices], F.normalize(rel_linear[indices])], dim=1)))

    assert len(right_neibor_emd_i) == len(left_neibor_emd_i)
    error_list = []
    iyu = 0
    for it in range(len(left_neibor_emd_i)):
        eta = 0.8 + (it / len(left_neibor_emd_i)) * 0.1
        rank_top = int(6 - 3 * (it / len(left_neibor_emd_i)))
        if img_features is not None:
            similar_matrix = left_neibor_emd_i[it].mm(right_neibor_emd_i[it].t()) * alpha + beta * left_neibor_emd_a[it].mm(
                right_neibor_emd_a[it].t()) + gamma * left_neibor_emd_r[it].mm(right_neibor_emd_r[it].t())
        else:
            similar_matrix = beta * left_neibor_emd_a[it].mm(
                right_neibor_emd_a[it].t()) + gamma * left_neibor_emd_r[it].mm(right_neibor_emd_r[it].t())
        num_elements = similar_matrix.numel()
        if num_elements < rank_top:
            top_k = 1
        else:
            top_k = rank_top
        max_value = torch.max(similar_matrix)
        if max_value > eta:
            values, indices = torch.topk(similar_matrix.view(-1), top_k)
            for i in range(top_k):
                if values[i] > eta:
                    left_i, right_i = indices[i] // similar_matrix.shape[1], indices[i] % similar_matrix.shape[1]
                    new_data = np.array([list(left_neibor[it])[left_i], list(right_neibor[it])[right_i]]).reshape(1, 2)
                    new_data_flat = new_data.flatten()
                    train_ill_flat = train_ill.flatten()

                    existing_elements = np.isin(new_data_flat, train_ill_flat)

                    if not existing_elements.any():
                        train_ill = np.vstack((train_ill, new_data))
                    else:
                        iyu += 1


        else:
            error_list.append(it)
    incorrect_rows = test(train_ill[:, 0], train_ill[:, 1], img_features, attr_features, rel_features, ills)
    train_ill = np.delete(train_ill, incorrect_rows, axis=0)
    visual_links = [tuple(row) for row in train_ill.tolist()]
    test_judge(visual_links,ills)

    with open(f'{args.cluster_num}_{args.data_split}_{args.unsup_k}_3.pkl', 'wb') as file:
        pickle.dump(train_ill, file)
        print("stage 3 is over!!")

    return train_ill