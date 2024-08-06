import torch
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random
from prompt_graph.utils import mkdir, graph_views
from prompt_graph.data import load4node, load4graph, NodePretrain
from torch.optim import Adam
import os
from.base import PreTrain


class GraphCL(PreTrain):
    r"""
        Inherited from PreTrain, forming the GraphCL pre-train method.
        It optimizes the model by maximizing the similarity of
        positive sample pairs and minimizing the similarity of negative sample pairs.
        See `here <https://arxiv.org/abs/2010.13902>`__ for more information.

        Args:
            *args (tuple): Additional attributes.
            **kwargs (dict): Additional attributes.
        """

    def __init__(self, *args, **kwargs):  # hid_dim=16
        super().__init__(*args, **kwargs)
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(self.hid_dim, self.hid_dim)).to(self.device)

    def load_graph_data(self):
        r"""Data loading."""
        if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora', 'Computers', 'Photo', 'Reddit', 'WikiCS', 'Flickr']:
            self.graph_list, self.input_dim = NodePretrain(dataname=self.dataset_name, num_parts=200)
        else:
            self.input_dim, self.out_dim, self.graph_list = load4graph(self.dataset_name, pretrained=True)

    def get_loader(self, graph_list, batch_size, aug1=None, aug2=None, aug_ratio=None):
        r"""
        Gets two data loaders, loader1 and loader2,
        which are used to load the graph data after different data enhancement operations.
        There are 3 types of data enhancement methods.
        :obj:`dropN`: randomly delete some nodes of from the graph;
        :obj:`permE`: randomly permutate some edges in the graph;
        :obj:`maskN`: randomly mask some node features in the graph.

        Args:
            graph_list (list): The original graph list.
            batch_size (int): The size of one batch.
            aug1 (str): The type of the first data enhancement operation can be selected as :obj:`dropN`, :obj:`permE`, or :obj:`maskN`
                 (default: :obj:`None`).
            aug2 (str): The type of the second data enhancement operation can be selected as :obj:`dropN`, :obj:`permE`, or :obj:`maskN`
                 (default: :obj:`None`).
            aug_ratio (float): Scale factor of the data enhancement operation. The value ranges from :obj:`0.1`, :obj:`0.2`, or :obj:`0.3`
                 (default: :obj:`None`).
        """

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")

        shuffle(graph_list)
        if aug1 is None:
            aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug2 is None:
            aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug_ratio is None:
            aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

        print("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

        view_list_1 = []
        view_list_2 = []
        for g in graph_list:
            view_g = graph_views(data=g, aug=aug1, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_1.append(view_g)
            view_g = graph_views(data=g, aug=aug2, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_2.append(view_g)

        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                             num_workers=1)  # you must set shuffle=False !
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                             num_workers=1)  # you must set shuffle=False !

        return loader1, loader2

    def forward_cl(self, x, edge_index, batch):
        r"""
        Forward process.

        Args:
            x (Tensor): The input tensor for operation.
            edge_index (Tensor): The index of the edges.
            batch (Tensor): Batch information.
        """
        x = self.gnn(x, edge_index, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        r"""The loss function.

        Args:
            x1 (Tensor): The tensor used for calculating similarity loss.
            x2 (Tensor): Same as x1.

        """
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = - torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-4)).mean()
        # loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        # loss = - torch.log(loss).mean()
        return loss

    def train_graphcl(self, loader1, loader2, optimizer):
        r"""Trains one time, using 2 loaders,
        and returns the average(of batch) loss for the training.


        Args:
            loader1 (DataLoader): Dataloader for training.
            loader2 (DataLoader): Dataloader for training.
            optimizer (Optimizer): The selected optimizer.
        """
        self.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(zip(loader1, loader2)):
            batch1, batch2 = batch
            optimizer.zero_grad()
            x1 = self.forward_cl(batch1.x.to(self.device), batch1.edge_index.to(self.device),
                                 batch1.batch.to(self.device))
            x2 = self.forward_cl(batch2.x.to(self.device), batch2.edge_index.to(self.device),
                                 batch2.batch.to(self.device))
            loss = self.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None, lr=0.01, decay=0.0001, epochs=100):
        r"""Performs multiple rounds of pre-training
        and saves the model with the least training loss at the end of each round.
        The pre-training effect of the model is gradually optimized through iterative loops
        and saved in the specified folder"""
        self.to(self.device)
        loader1, loader2 = self.get_loader(self.graph_list, batch_size, aug1=aug1, aug2=aug2)
        print('start training {} | {} | {}...'.format(self.dataset_name, 'GraphCL', self.gnn_type))
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        for epoch in range(1, epochs + 1):  # 1..100
            train_loss = self.train_graphcl(loader1, loader2, optimizer)

            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                torch.save(self.gnn.state_dict(),
                           "./Experiment/pre_trained_model/{}/{}.{}.{}.pth".format(self.dataset_name, 'GraphCL',
                                                                                   self.gnn_type,
                                                                                   str(self.hid_dim) + 'hidden_dim'))
                print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'GraphCL', self.gnn_type,
                                                                str(self.hid_dim) + 'hidden_dim'))
