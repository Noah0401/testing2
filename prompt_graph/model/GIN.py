import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch, gc
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv,GINConv,SAGEConv
from torch_geometric.nn import GraphConv as GConv
import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as skm
from prompt_graph.utils import act


class GIN(torch.nn.Module):
    r"""
    Inherited from :class:`torch.nn.Module`, forming a GIN model;
    GIN (Graph Isomorphism Network) learns the representation of nodes by aggregating
    its neighbors, see `here <https://arxiv.org/abs/1810.00826>`__ for more information.

        Args:
            input_dim (int): the dimension of the input node feature
            hid_dim (int): the dimension of the hidden layer (default: :obj:`None`)
            out_dim (int): the dimension of output (default: :obj:`None`)
            num_layer (int): the number of GNN layers (default: :obj:`3`)
            JK (str): last, concat, max or sum. (default: :obj:`last`)
            drop_ratio (float): dropout rate (default: :obj:`0`)
            pool (str): sum, mean, max, attention, set2set (default: :obj:`mean`)
        """

    def __init__(self, input_dim, hid_dim=None, out_dim=None, num_layer=3, JK="last", drop_ratio=0, pool='mean'):
        super().__init__()
        GraphConv = lambda i, h: GINConv(nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)))

        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if num_layer < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(num_layer))
        elif num_layer == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        self.JK = JK
        self.drop_ratio = drop_ratio
        # Different kind of graph pooling
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        # elif pool == "attention":
        #     self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, x, edge_index, batch=None, prompt=None, prompt_type=None):
        h_list = [x]
        for idx, conv in enumerate(self.conv_layers[0:-1]):
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.conv_layers[-1](x, edge_index)
        h_list.append(x)
        if self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]

        if batch == None:
            return node_emb
        else:
            if prompt_type == 'Gprompt':
                node_emb = prompt(node_emb)
            graph_emb = self.pool(node_emb, batch.long())
            return graph_emb

    def decode(self, z, edge_label_index):
        r"""Computes the decoding results of the given node embedding and edge label index.

        Args:
            z (Tensor): Used to compute the decoding result of a given node embedding and edge label index.
            edge_label_index (Tensor): The index of the labels of edges.
        """
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        r"""Computes the decoding result matrix embedded by a given node.

        Args:
            z (Tensor): Used to compute the decoding result of a given node embedding and edge label index.
        """
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()