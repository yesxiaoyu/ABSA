# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        # support = torch.matmul(text, self.weight)
        # output = torch.matmul(adj, support)

        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # self.a = nn.Parameter(torch.FloatTensor(out_features))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, text, adj):
        # print('W.shape', self.W.shape)
        # print('input.shape', text.shape)
        B, N, C = text.size()
        # print('B', B)
        # print('N', N)
        # print('C', C)
        h = torch.matmul(text, self.W)
        # N = h.size()[0]
        # print('h.shape', h.shape)
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, C), h.repeat(1, N, 1)], dim=2).view(B, N, N,
                                                                                        2 * self.out_features)  # [B,N,N,2C]

        attention = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # [B,N,N]

        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2) # [B,N,N]
        attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.matmul(attention, h)
        #
        h_prime = torch.bmm(attention, h)  # [B,N,N]*[B,N,C]-> [B,N,C]
        # if self.concat:
        #     return F.elu(h_prime)
        # else:
        #     return h_prime
        # denom = torch.sum(adj, dim=2, keepdim=True) + 1
        # if self.concat:
        #     return F.elu(h_prime + self.alpha * h)#/denom
        # else:
        #     return (h_prime + self.alpha * h)# /denom


        output = F.elu(h_prime + self.alpha * h)
        # # output = out / denom
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, embedding_matrix, opt, dropout=0.6, alpha=0.2, nheads=6):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.opt = opt

        # nfeat, nhid
        self.attentions = [GraphAttentionLayer(2*opt.hidden_dim, 2*opt.hidden_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(2*opt.hidden_dim, 2*opt.hidden_dim, dropout=dropout, alpha=alpha, concat=False)
        self.text_embed_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)

    # def normalize_adj(self, adj, threshold=0.1):
    #     adj = (adj > threshold) * adj
    #     # adj[adj>0.4]=1
    #     D = np.power(np.sum(adj, axis=1), -0.5)
    #     D = np.diag(D)
    #     adj = np.matmul(np.matmul(D, adj), D)
    #     return adj

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs

        # hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        # output = torch.matmul(adj, hidden) / denom

        # adj = self.normalize_adj(adjs)
        adj /= denom

        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)

        text_out, (_, _) = self.text_lstm(text, text_len)

        x = F.dropout(text_out, self.dropout, training=self.training)
        # self.position_weight(text_out, aspect_double_idx, text_len, aspect_len)
        # x = F.dropout(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), self.dropout, training=self.training)

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj) # / denom

        # x = text_out

        x = self.mask(x, aspect_double_idx)
        # return F.log_softmax(x, dim=1)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output
# GraphConvolution  GraphAttentionLayer
class ASGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        # print('self.gc1.shape', self.gc1.shape)
        # self.conv1 = nn.Conv1d(2*opt.hidden_dim, 2*opt.hidden_dim, 3, padding=1)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        # print('self.gc2.shape', self.gc2.shape)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        # print('self.fc.shape', self.fc.shape)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        # x = F.relu(self.gc1(text_out, adj))
        # x = F.relu(self.gc2(x, adj))

        x = self.mask(x, aspect_double_idx)

        # x = F.relu(self.conv1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len).transpose(1, 2)))
        # x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        # x = self.mask(x.transpose(1, 2), aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output