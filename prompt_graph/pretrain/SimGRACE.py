import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from prompt_graph.utils import mkdir
from torch.optim import Adam
from prompt_graph.data import load4node, load4graph, NodePretrain
from copy import deepcopy
from.base import PreTrain
import os


class SimGRACE(PreTrain):
    r"""
        Inherited from PreTrain, forming the SimGRACE pre-train method;
        By using different graph encoders as comparison view generators,
        the semantic similarity between the views obtained from two different encoders is compared;
        By narrowing the vector representation of the same semantics in vector space,
        contrast learning is carried out;
        See `here <https://arxiv.org/abs/2202.03104>`__ for more information.

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

    def get_loader(self, graph_list, batch_size):
        r"""Gets the data loader.

        Args:
            graph_list (list): The list of graphs.
            batch_size (int): The size of a batch.

        """
        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in SimGRACE!")

        loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=1)
        return loader

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

    def perturbate_gnn(self, data):
        r"""Perturbation of GNN model parameters.

        Args:
            data (Data): The input graph data.
        """
        vice_model = deepcopy(self).to(self.device)

        for (vice_name, vice_model_param) in vice_model.named_parameters():
            if vice_name.split('.')[0] != 'projection_head':
                std = vice_model_param.data.std() if vice_model_param.data.numel() > 1 else torch.tensor(1.0)
                noise = 0.1 * torch.normal(0, torch.ones_like(vice_model_param.data) * std)
                vice_model_param.data += noise
        z2 = vice_model.forward_cl(data.x, data.edge_index, data.batch)
        return z2

    def train_simgrace(self, loader, optimizer):
        r"""Trains for one time,
            returns the average(of batch) loss for the training.

            Args:
                loader (DataLoader): Dataloader for training.
                optimizer (Optimizer): The selected optimizer.
                """
        self.train()
        train_loss_accum = 0
        total_step = 0
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            x2 = self.perturbate_gnn(data)
            x1 = self.forward_cl(data.x, data.edge_index, data.batch)
            x2 = Variable(x2.detach().data.to(self.device), requires_grad=False)
            loss = self.loss_cl(x1, x2)
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, lr=0.01, decay=0.0001, epochs=100):

        r"""Perform multiple rounds of pre-training
            and save the model with the least training loss at the end of each round.
            The pre-training effect of the model is gradually optimized through iterative loops
            and saved in the specified folder.

            Args:
                batch_size (int): The size of a batch (default: :obj:`10`).
                lr (float): Learning rate which controls how much the model parameters are adjusted at each iteration (default: :obj:`0.01`).
                decay (float): Weight decay (default: :obj:`0.0001`).
                epochs (int): How many times the training process will be done (default: :obj:`100`).
            """

        loader = self.get_loader(self.graph_list, batch_size)
        print('start training {} | {} | {}...'.format(self.dataset_name, 'SimGRACE', self.gnn_type))
        optimizer = optim.Adam(self.gnn.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        for epoch in range(1, epochs + 1):  # 1..100
            train_loss = self.train_simgrace(loader, optimizer)

            print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                torch.save(self.gnn.state_dict(),
                           "./Experiment/pre_trained_model/{}/{}.{}.{}.pth".format(self.dataset_name, 'SimGRACE',
                                                                                   self.gnn_type,
                                                                                   str(self.hid_dim) + 'hidden_dim'))
                print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'SimGRACE', self.gnn_type,
                                                                str(self.hid_dim) + 'hidden_dim'))
