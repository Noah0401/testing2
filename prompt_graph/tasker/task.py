import torch
from prompt_graph.model import GAT, GCN, GCov, GIN, GraphSAGE, GraphTransformer
from prompt_graph.prompt import GPF, GPF_plus, LightPrompt,HeavyPrompt, Gprompt, GPPTPrompt, DiffPoolPrompt, SAGPoolPrompt
from prompt_graph.prompt import featureprompt, downprompt
from prompt_graph.pretrain import PrePrompt
from torch import nn, optim
from prompt_graph.data import load4node, load4graph
from prompt_graph.utils import Gprompt_tuning_loss
import numpy as np


class BaseTask:
    r"""
    Being the base class for graph, link, node tasks;
    Initialize the optimizer, loss function, prompt, and GNN.

    Args:
        pre_train_model_path (str): The path of the pre-train model. (default: :obj:`None`)
        gnn_type (str): The type of GNN. (default: :obj:`TransformerConv`)
        hid_dim (int): The dimension of hidden layers. (default: :obj:`128`)
        num_layer (int): The number of layers. (default: :obj:`2`)
        dataset_name (str): The name of the dataset (graphs). (default: :obj:`Cora`)
        prompt_type (str): The type of Prompt. (default: :obj:`GPF`)
        epochs (int): The training rounds. (default: :obj:`100`)
        shot_num (int): The number of shots. The number of samples chosen in each class (default: :obj:`10`)
        device (int): The chosen device. (default: :obj:`5`)

    """

    def __init__(self, pre_train_model_path=None, gnn_type='TransformerConv', hid_dim=128, num_layer=2,
                 dataset_name='Cora',
                 prompt_type='GPF', epochs=100, shot_num=10, device: int = 5):
        self.pre_train_model_path = pre_train_model_path
        self.device = torch.device('cuda:' + str(device) if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.num_layer = num_layer
        self.dataset_name = dataset_name
        self.shot_num = shot_num
        self.gnn_type = gnn_type
        self.prompt_type = prompt_type
        self.epochs = epochs
        self.initialize_lossfn()

    def initialize_optimizer(self):
        r"""
        Optimizer initialization.
         The optimizer is chosen and configured differently depending
         on the prompt type specified in the constructor.
        """
        if self.prompt_type == 'None':
            model_param_group = []
            model_param_group.append({"params": self.gnn.parameters()})
            model_param_group.append({"params": self.answering.parameters()})
            self.optimizer = optim.Adam(model_param_group, lr=0.005, weight_decay=5e-4)
        elif self.prompt_type == 'All-in-one':
            self.pg_opi = optim.Adam(filter(lambda p: p.requires_grad, self.prompt.parameters()), lr=0.001,
                                     weight_decay=0.00001)
            self.answer_opi = optim.Adam(filter(lambda p: p.requires_grad, self.answering.parameters()), lr=0.001,
                                         weight_decay=0.00001)
        elif self.prompt_type in ['GPF', 'GPF-plus']:
            model_param_group = []
            model_param_group.append({"params": self.prompt.parameters()})
            model_param_group.append({"params": self.answering.parameters()})
            self.optimizer = optim.Adam(model_param_group, lr=0.005, weight_decay=5e-4)
        elif self.prompt_type in ['Gprompt', 'GPPT']:
            self.pg_opi = optim.Adam(self.prompt.parameters(), lr=0.01, weight_decay=5e-4)
        elif self.prompt_type == 'MultiGprompt':
            self.optimizer = torch.optim.Adam([*self.DownPrompt.parameters(), *self.feature_prompt.parameters()],
                                              lr=0.001)

    def initialize_lossfn(self):
        r"""
            Initializes the loss function based on the prompt type.
            The loss function is set to the cross-entropy loss by default, but for the :obj:`Gprompt` prompt type,
            a custom loss function called :obj:`Gprompt_tuning_loss` is used.
        """
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.prompt_type == 'Gprompt':
            self.criterion = Gprompt_tuning_loss()

    def initialize_prompt(self):
        r"""
        Initializes the prompt based on the prompt type.
        The prompt is a component used in the task and its initialization varies depending on the prompt type
        specified in the constructor.
        """
        if self.prompt_type == 'None':
            self.prompt = None
        elif self.prompt_type == 'GPPT':
            if (self.task_type == 'NodeTask'):
                self.prompt = GPPTPrompt(self.hid_dim, self.output_dim, self.output_dim, device=self.device)
                train_ids = torch.nonzero(self.data.train_mask, as_tuple=False).squeeze()
                node_embedding = self.gnn(self.data.x, self.data.edge_index)
                self.prompt.weigth_init(node_embedding, self.data.edge_index, self.data.y, train_ids)
            elif (self.task_type == 'GraphTask'):
                self.prompt = GPPTPrompt(self.hid_dim, self.output_dim, self.output_dim, device=self.device)
        elif self.prompt_type == 'All-in-one':
            lr, wd = 0.001, 0.00001
            # self.prompt = LightPrompt(token_dim=self.input_dim, token_num_per_group=100, group_num=self.output_dim, inner_prune=0.01).to(self.device)
            self.prompt = HeavyPrompt(token_dim=self.input_dim, token_num=10, cross_prune=0.1, inner_prune=0.3).to(
                self.device)
        elif self.prompt_type == 'GPF':
            self.prompt = GPF(self.input_dim).to(self.device)
        elif self.prompt_type == 'GPF-plus':
            self.prompt = GPF_plus(self.input_dim, 20).to(self.device)
        elif self.prompt_type == 'sagpool':
            self.prompt = SAGPoolPrompt(self.input_dim, num_clusters=5, ratio=0.5).to(self.device)
        elif self.prompt_type == 'diffpool':
            self.prompt = DiffPoolPrompt(self.input_dim, num_clusters=5).to(self.device)
        elif self.prompt_type == 'Gprompt':
            self.prompt = Gprompt(self.hid_dim).to(self.device)
        elif self.prompt_type == 'MultiGprompt':
            nonlinearity = 'prelu'
            self.Preprompt = PrePrompt(self.dataset_name, self.hid_dim, nonlinearity, 0.9, 0.9, 0.1, 0.001, 1, 0.3).to(
                self.device)
            self.Preprompt.load_state_dict(torch.load(self.pre_train_model_path))
            self.Preprompt.eval()
            self.feature_prompt = featureprompt(self.Preprompt.dgiprompt.prompt,
                                                self.Preprompt.graphcledgeprompt.prompt,
                                                self.Preprompt.lpprompt.prompt).to(self.device)
            dgiprompt = self.Preprompt.dgi.prompt
            graphcledgeprompt = self.Preprompt.graphcledge.prompt
            lpprompt = self.Preprompt.lp.prompt
            self.DownPrompt = downprompt(dgiprompt, graphcledgeprompt, lpprompt, 0.001, self.hid_dim, 7,
                                         self.device).to(self.device)
        else:
            raise KeyError(" We don't support this kind of prompt.")

    def initialize_gnn(self):
        r"""
        Initializes GNN based on the specified GNN type.
        """
        if self.gnn_type == 'GAT':
            self.gnn = GAT(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCN':
            self.gnn = GCN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphSAGE':
            self.gnn = GraphSAGE(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GIN':
            self.gnn = GIN(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GCov':
            self.gnn = GCov(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        elif self.gnn_type == 'GraphTransformer':
            self.gnn = GraphTransformer(input_dim=self.input_dim, hid_dim=self.hid_dim, num_layer=self.num_layer)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        self.gnn.to(self.device)

        if self.pre_train_model_path != 'None' and self.prompt_type != 'MultiGprompt':
            if self.gnn_type not in self.pre_train_model_path:
                raise ValueError(f"the Downstream gnn '{self.gnn_type}' does not match the pre-train model")
            if self.dataset_name not in self.pre_train_model_path:
                raise ValueError(f"the Downstream dataset '{self.dataset_name}' does not match the pre-train dataset")

            self.gnn.load_state_dict(torch.load(self.pre_train_model_path, map_location=self.device))
            print("Successfully loaded pre-trained weights!")