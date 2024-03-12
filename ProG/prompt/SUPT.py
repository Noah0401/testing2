from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.init import normal_ as normal
from torch.nn.init import orthogonal_ as orthogonal
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import GATConv, GINConv, GCNConv, SGConv

from ProG.data.pooling import TopKPooling, SAGPooling #, MemPooling

from torch_geometric.nn import dense_diff_pool, DenseGINConv, DenseGCNConv, DenseSAGEConv

from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential
import pdb

from torch import Tensor
from torch_geometric.typing import PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
    to_dense_adj,
    to_dense_batch,
    select,
    softmax,
    scatter,
)



class SAGPoolPrompt(nn.Module):
    def __init__(self, in_channels: int, num_clusters=1, ratio=0.5, orth_loss=False, softmax=False, nonlinearity='tanh'):
        super(SAGPoolPrompt, self).__init__()
        # self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))

        self.pools = nn.ModuleList()
        self.pools_emb = nn.Parameter(torch.Tensor(num_clusters, in_channels))
        self.num_clusters = num_clusters
        
        # self.gnn = SGConv(in_channels, 4, K=2)
        # self.linear = nn.Linear(4, num_clusters)
        
        self.gnn = GCNConv(in_channels, num_clusters)

        if isinstance(nonlinearity, str):
            self.nonlinearity = getattr(torch, nonlinearity)
        self.ratio = ratio
        self.softmax = softmax

        self.orth_loss = orth_loss
        if orth_loss:
            self.eye = torch.eye(num_clusters).cuda()
            self.eye.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()
        # self.linear.reset_parameters()
        if self.num_clusters>1 and self.orth_loss:
            orthogonal(self.pools_emb)
        else:
            # normal(self.pools_emb)
            glorot(self.pools_emb)
        # glorot(self.global_emb)
        # normal(self.global_emb)

    def orthogonal_loss(self):
        return torch.norm(torch.mm(self.pools_emb, self.pools_emb.t()) - self.eye, p=2)

    def add(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        p = torch.zeros_like(x)
        # x = x+self.global_emb
        # score = self.linear(torch.relu(self.gnn(x+self.pools_emb.sum(dim=0), edge_index)))
        score = self.gnn(x+self.pools_emb.sum(dim=0), edge_index)

        if self.softmax:
            score = softmax(score, batch)
        else:
            score = self.nonlinearity(score)
        
        idx_count = torch.ones_like(batch)
        for i in range(self.num_clusters):
            perm = topk(score[:,i], self.ratio, batch, None)
            p_ = score[:,i][perm].unsqueeze(1)*self.pools_emb[i]
            if self.orth_loss: p_ = p_/self.num_clusters
            idx_count[perm] += 1
            p[perm] += p_ 
        # pdb.set_trace()
        # print((p/idx_count.unsqueeze(1)).max())
        return x + p/idx_count.unsqueeze(1) #+ self.global_emb



class DiffPoolPrompt(nn.Module):
    def __init__(self, in_channels: int, num_clusters=10, orth_loss=False):
        super(DiffPoolPrompt, self).__init__()
        # if num_clusters==1:
        #     assert False, "DiffPoolPrompt should have num_clusters>1"

        # self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.cluster_emb = nn.Parameter(torch.Tensor(num_clusters, in_channels))

        self.num_clusters = num_clusters

        # self.gnn_pool = SGConv(in_channels, 4, K=2)
        # self.linear = nn.Linear(4, num_clusters)

        self.gnn_pool = GCNConv(in_channels, num_clusters)

        # self.gnn_pool = DenseGCNConv(in_channels, num_clusters)
        # self.gnn_pool = DenseSAGEConv(in_channels, num_clusters)
        self.link_loss = 0
        self.ent_loss = 0
        self.orth_loss = orth_loss
        if orth_loss:
            self.eye = torch.eye(num_clusters).cuda()
            self.eye.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn_pool.reset_parameters()
        # self.linear.reset_parameters()
        # glorot(self.global_emb)
        if self.num_clusters>1 and self.orth_loss:
            orthogonal(self.cluster_emb)
        else:
            # normal(self.pools_emb)
            glorot(self.cluster_emb)
        
    def orthogonal_loss(self):
        return torch.norm(torch.mm(self.cluster_emb, self.cluster_emb.t()) - self.eye, p=2)
    
    def add(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        # x = x*self.global_emb

        # s = softmax(self.gnn_pool(x+self.cluster_emb.sum(dim=0), edge_index),batch)

        # s = self.linear(torch.relu(torch.softmax(self.gnn_pool(x+self.cluster_emb.sum(dim=0), edge_index), dim=1)))
        s = torch.softmax(self.gnn_pool(x+self.cluster_emb.sum(dim=0), edge_index), dim=1)

        # pdb.set_trace()
        
        # top_s_score, top_s_idx = torch.topk(s, 2, dim=1)
        # selected = self.cluster_emb[top_s_idx]
        # p = (selected*top_s_score.unsqueeze(-1)).sum(dim=1)

        p = torch.matmul(s, self.cluster_emb)

        # if self.training:
        #     self.ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()
            # s_out, _ = to_dense_batch(s, batch)
            # adj = to_dense_adj(edge_index, batch)
            # self.link_loss = torch.norm(adj-torch.matmul(s_out, s_out.transpose(1, 2)), p=2) / adj.numel()
        

        return x + p #+ self.global_emb