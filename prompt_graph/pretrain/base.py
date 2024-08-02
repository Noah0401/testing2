import torch
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from torch.optim import Adam


class PreTrain(torch.nn.Module):
    r"""Inherited from :class:`torch.nn.Module`, being the base class for concrete pre-train method.
        Initialize the GNN model to prepare for the training.

        Args:
            gnn_type (str): The type of GNN.
                (default: :obj:`TransformerConv`)
            dataset_name (str): The name of the dataset.
                (default: :obj:`Cora`)
            hid_dim (int): The dimensionality of the hidden layers.
                (default: :obj:`128`)
            gln (int): The number of layers in the GNN.
                (default: :obj:`2`)
            num_epoch (int): The number of training epochs.
                (default: :obj:`100`)
            device (int):  A PyTorch torch.device object that represents the training device.
                (default: :obj:`5`)
        """

    def __init__(self, gnn_type='TransformerConv', dataset_name='Cora', hid_dim=128, gln=2, num_epoch=100,
                 device: int = 5):
        super().__init__()
        self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.gnn_type = gnn_type
        self.num_layer = gln
        self.epochs = num_epoch
        self.hid_dim = hid_dim

    def initialize_gnn(self, input_dim, hid_dim):
        r"""Initialize the GNN model"""
        if self.gnn_type == 'GAT':
            self.gnn = GAT(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCN':
            self.gnn = GCN(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
            self.gnn = GraphSAGE(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GIN':
            self.gnn = GIN(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCov':
            self.gnn = GCov(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphTransformer':
            self.gnn = GraphTransformer(input_dim=input_dim, hid_dim=hid_dim, num_layer=self.num_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        print(self.gnn)
        self.gnn.to(self.device)
        self.optimizer = Adam(self.gnn.parameters(), lr=0.001, weight_decay=0.00005)

#     def load_node_data(self):
#         self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
#         self.data.to(self.device)
#         self.input_dim = self.dataset.num_features
#         self.output_dim = self.dataset.num_classes

