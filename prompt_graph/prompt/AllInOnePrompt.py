import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from prompt_graph.utils import act
from deprecated.sphinx import deprecated
from sklearn.cluster import KMeans
from torch_geometric.nn.inits import glorot


class LightPrompt(torch.nn.Module):

    r"""
    Inherited from :class:`torch.nn.Module`;
    Initializes all the tokens and
    generates the initial prompt graph by combining all the tokens.

    Args:
        token_dim (int): The dimension of the token.
        token_num_per_group (int): The number of tokens in one group.
        group_num (int):  The total token number = token_num_per_group*group_num, in most cases, we let group_num=:obj:`1`.
                    In :obj:`prompt_w_o_h` mode for classification, we can let each class correspond to one group.
                    You can also assign each group as a prompt batch in some cases
                    (default: :obj:`1`).
        inner_prune (int/float): The threshold of pruning operations to determine which connections should be cut.
                    If :obj:`inner_prune` is not :obj:`None`, then cross prune adopt :obj:`prune_thre` whereas inner prune adopt :obj:`inner_prune`
                    (default: :obj:`None`).
    """
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):

        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        r"""Uses :obj:`Kaiming uniform` method to initialize token;
        It is a method that initializes the weights of a neural network layer with
        values drawn from a uniform distribution;
        This initialization is specifically designed for layers that use rectified
        linear units (ReLU) as activation functions.

        Args:
            init_method (str): They way been chosen (default: :obj:`kaiming_uniform`).
        """
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self) -> Batch:
        r"""Returns the batch of prompt graphs."""
        return self.token_view()

    def token_view(self, ) -> Batch:
        r"""
        Each token group is viewed as a prompt sub-graph, and
        turns all groups of tokens as a batch of prompt graphs.
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch


class HeavyPrompt(LightPrompt):
    r"""
    Inherited from :class:`LightPrompt`, forming a prompt from tokens
    using cross-group pruning and internal pruning methods.

    Args:
        token_dim (int): The dimension of the tokne.
        token_num (int): The number of the tokens.
        cross_prune (float): The threshold of pruning operations to determine which crossing connections should be cut.
        (default: :obj:`0.1`)
        inner_prone (float): The threshold of pruning operations to determine which inner connections should be cut.
        (default: :obj:`0.01`)
    """

    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, graph_batch: Batch):

        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num

            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)

            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num

            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch

    def Tune(self, train_loader, gnn, answering, lossfn, opi, device):
        r"""
        Doing fine-tune of the Prompt. Using :obj:`answering` to calculate the loss.

        Args:
            train_loader (DataLoader): The chosen training dataloader.
            gnn (model): The chosen GNN model.
            answering (model): The predicted answer.
            lossfn (function): The chosen loss function.
            opi (Optimizer): The chosen optimizer.
            device (Device): The used device.
        """
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader):
            # print(train_batch)
            train_batch = train_batch.to(device)
            prompted_graph = self.forward(train_batch)
            # print(prompted_graph)

            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
            train_loss = lossfn(pre, train_batch.y)

            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()

        return running_loss / len(train_loader)

    def TuneWithoutAnswering(self, train_loader, gnn, answering, lossfn, opi, device):
        r"""
            Doing fine-tune of the Prompt. Without Using :obj:`answering` to calculate the loss.

            Args:
            train_loader (DataLoader): The chosen training dataloader.
            gnn (model): The chosen GNN model.
            answering (model,Optional): The predicted answer.
            lossfn (function): The chosen loss function.
            opi (Optimizer,Optional): The chosen optimizer.
            device (Device,Optional): The used device.
        """
        total_loss = 0.0
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            emb0 = gnn(batch.x, batch.edge_index, batch.batch)
            pg_batch = self.inner_structure_update()
            pg_batch = pg_batch.to(self.device)
            pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            # cross link between prompt and input graphs
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
            sim = torch.softmax(dot, dim=1)
            loss = lossfn(sim, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)


class FrontAndHead(torch.nn.Module):
    r""" Inherit from :class:`torch.nn.Module`;
    After the input graph data is prompted by :class:`HeavyPrompt`,
    the GNN model is used to extract the feature of the prompted graph data,
    and finally the prediction is carried out by the full connection layer and softmax function;
    This class is used in multi-label classification tasks
    and provides an example of building front-end and header models in a context that uses :class:`HeavyPrompt`.

    Args:
        input_dim (int): The dimension of input.
        hid_dim (int): The dimension of hidden layer. (default: :obj:`16`)
        num_classes (int): The number of classes. (default: :obj:`2`)
        task_type (str): The type of the task. (default: :obj:`multi_label_classification`)
        token_num (int): The number of the tokens (default: :obj:`10`).
        cross_prune (float): The threshold of pruning operations to determine which crossing connections should be cut.
            (default: :obj:`0.1`)
        inner_prone (float): The threshold of pruning operations to determine which inner connections should be cut.
            (default: :obj:`0.3`)

    """

    def __init__(self, input_dim, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3):

        super().__init__()

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch, gnn):
        prompted_graph = self.PG(graph_batch)
        graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pre



