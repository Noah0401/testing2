import os
import torch
import random
import networkx as nx
import pandas as pd
import numpy as np
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
from collections import Counter, deque
from networkx.algorithms.traversal.breadth_first_search import generic_bfs_edges


import requests

url = "https://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip"
response = requests.get(url)
with open("dataset.zip", "wb") as f:
    f.write(response.content)

def nx_to_graph_data_obj(g:nx, center_id:any, allowable_features_downstream:list=None,
                         allowable_features_pretrain:list=None,
                         node_id_to_go_labels:dict=None)->Data:
    r"""Converts nx graph of PPI to pytorch geometric Data object;
    Returns :class:`torch_geometric.data.Data` object with the following attributes:
    :obj:`edge_attr`, :obj:`edge_index`, :obj:`x`
    , :obj:`species_id`, :obj:`center_node_idx`
    , :obj:`go_target_downstream` (only if node_id_to_go_labels is not None)
    , :obj:`go_target_pretrain` (only if node_id_to_go_labels is not None).

        Args:
            g (nx): nx graph object of ego graph
            center_id (Any): node id of center node in the ego graph
            allowable_features_downstream (list): list of possible go function node
                features for the downstream task. The resulting go_target_downstream node
                feature vector will be in this order.
            allowable_features_pretrain (list): list of possible go function node
                features for the pretraining task. The resulting go_target_pretrain node
                feature vector will be in this order.
            node_id_to_go_labels (dict): dict that maps node id to a list of its
                corresponding go labels
    """

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    # nodes
    nx_node_ids = [n_i for n_i in g.nodes()]  # contains list of nx node ids
    # in a particular ordering. Will be used as a mapping to convert
    # between nx node ids and data obj node indices

    x = torch.tensor(np.ones(n_nodes).reshape(-1, 1), dtype=torch.float)
    # we don't have any node labels, so set to dummy 1. dim n_nodes x 1

    center_node_idx = nx_node_ids.index(center_id)
    center_node_idx = torch.tensor([center_node_idx], dtype=torch.long)

    # edges
    edges_list = []
    edge_features_list = []
    for node_1, node_2, attr_dict in g.edges(data=True):
        edge_feature = [attr_dict['w1'], attr_dict['w2'], attr_dict['w3'],
                        attr_dict['w4'], attr_dict['w5'], attr_dict['w6'],
                        attr_dict['w7'], 0, 0]  # last 2 indicate self-loop
        # and masking
        edge_feature = np.array(edge_feature, dtype=int)
        # convert nx node ids to data obj node index
        i = nx_node_ids.index(node_1)
        j = nx_node_ids.index(node_2)
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        edges_list.append((j, i))
        edge_features_list.append(edge_feature)

    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

    # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    edge_attr = torch.tensor(np.array(edge_features_list),
                             dtype=torch.float)

    try:
        species_id = int(nx_node_ids[0].split('.')[0])  # nx node id is of the form:
        # species_id.protein_id
        species_id = torch.tensor([species_id], dtype=torch.long)
    except:  # occurs when nx node id has no species id info. For the extract
        # substructure context pair transform, where we convert a data obj to
        # a nx graph obj (which does not have original node id info)
        species_id = torch.tensor([0], dtype=torch.long)    # dummy species
        # id is 0

    # construct data obj
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.species_id = species_id
    data.center_node_idx = center_node_idx

    if node_id_to_go_labels:  # supervised case with go node labels
        # Construct a dim n_pretrain_go_classes tensor and a
        # n_downstream_go_classes tensor for the center node. 0 is no data
        # or negative, 1 is positive.
        downstream_go_node_feature = [0] * len(allowable_features_downstream)
        pretrain_go_node_feature = [0] * len(allowable_features_pretrain)
        if center_id in node_id_to_go_labels:
            go_labels = node_id_to_go_labels[center_id]
            # get indices of allowable_features_downstream that match with elements
            # in go_labels
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_downstream, go_labels, return_indices=True)
            for idx in node_feature_indices:
                downstream_go_node_feature[idx] = 1
            # get indices of allowable_features_pretrain that match with
            # elements in go_labels
            _, node_feature_indices, _ = np.intersect1d(
                allowable_features_pretrain, go_labels, return_indices=True)
            for idx in node_feature_indices:
                pretrain_go_node_feature[idx] = 1
        data.go_target_downstream = torch.tensor(np.array(downstream_go_node_feature),
                                        dtype=torch.long)
        data.go_target_pretrain = torch.tensor(np.array(pretrain_go_node_feature),
                                        dtype=torch.long)

    return data

def graph_data_obj_to_nx(data:Data)->nx:
    r"""Converts pytorch geometric Data obj to network x data object.

    Args:
        data (Data): The provided graph.
    """
    G = nx.Graph()

    # edges
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    n_edges = edge_index.shape[1]
    for j in range(0, n_edges, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        w1, w2, w3, w4, w5, w6, w7, _, _ = edge_attr[j].astype(bool)
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5,
                       w6=w6, w7=w7)

    # # add center node id information in final nx graph object
    # nx.set_node_attributes(G, {data.center_node_idx.item(): True}, 'is_centre')

    return G


class BioDataset(InMemoryDataset):
    r"""
        Inherited from :class:`torch_geometric.data.InMemoryDataset`, creating dataset of biography.
        See `here <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/in_memory_dataset.html#InMemoryDataset>`__ for the accompanying
        tutorial.

        Args:
            root (str): The data directory that contains a raw and processed dir.
            data_type (str): either supervised or unsupervised.
            empty (bool): if obj:`TRUE`, then will not load any data obj. For
            initializing empty dataset (default: :obj:`FALSE`).
            transform (callable, optional): A function/transform that takes in a
                :class:`~torch_geometric.data.Data` or
                :class:`~torch_geometric.data.HeteroData` object and returns a
                transformed version.
                The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                a :class:`~torch_geometric.data.Data` or
                :class:`~torch_geometric.data.HeteroData` object and returns a
                transformed version.
                The data object will be transformed before being saved to disk.
                (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in a
                :class:`~torch_geometric.data.Data` or
                :class:`~torch_geometric.data.HeteroData` object and returns a
                boolean value, indicating whether the data object should be
                included in the final dataset. (default: :obj:`None`)
    """
    def __init__(self,
                 root,
                 data_type,
                 empty=False,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        self.root = root
        self.data_type = data_type

        super(BioDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        #raise NotImplementedError('Data is assumed to be processed')
        if self.data_type == 'supervised': # 8 labelled species
            file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090', '4932', '7955']
        else: # unsupervised: 8 labelled species, and 42 top unlabelled species by n_nodes.
            file_name_list = ['3702', '6239', '511145', '7227', '9606', '10090',
            '4932', '7955', '3694', '39947', '10116', '443255', '9913', '13616',
            '3847', '4577', '8364', '9823', '9615', '9544', '9796', '3055', '7159',
            '9031', '7739', '395019', '88036', '9685', '9258', '9598', '485913',
            '44689', '9593', '7897', '31033', '749414', '59729', '536227', '4081',
            '8090', '9601', '749927', '13735', '448385', '457427', '3711', '479433',
            '479432', '28377', '9646']
        return file_name_list


    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        raise NotImplementedError('Data is assumed to be processed')

if __name__ == "__main__":
    


    root_supervised = 'dataset/supervised'

    d_supervised = BioDataset(root_supervised, data_type='supervised')

    print(d_supervised)

    root_unsupervised = 'dataset/unsupervised'
    d_unsupervised = BioDataset(root_unsupervised, data_type='unsupervised')

    print(d_unsupervised)


