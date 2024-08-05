from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import GraphConv
from torch_geometric.typing import OptTensor
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import scatter, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes


def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    r"""
    Selects the index corresponding to the first k maximum values in the input tensor x.

    Args:
        x (Tensor): The features/scores of the nodes.
        ratio (Union[float, int]): The selection ratio.
        batch (Tensor): The batch of the graph.
        min_score (Optional, float): Indicates the minimum score threshold. If :obj:`min_score` is specified,
        only nodes with scores greater than this threshold are selected.
        tol (float): Tolerance for handling floating-point comparisons.
    """

    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)

    elif ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -60000.0)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
            k = torch.min(k, num_nodes)
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        if isinstance(ratio, int) and (k == ratio).all():
            # If all graphs have exactly `ratio` or more than `ratio` entries,
            # we can just pick the first entries in `perm` batch-wise:
            index = torch.arange(batch_size, device=x.device) * max_num_nodes
            index = index.view(-1, 1).repeat(1, ratio).view(-1)
            index += torch.arange(ratio, device=x.device).repeat(batch_size)
        else:
            # Otherwise, compute indices per graph:
            index = torch.cat([
                torch.arange(k[i], device=x.device) + i * max_num_nodes
                for i in range(batch_size)
            ], dim=0)

        perm = perm[index]

    else:
        raise ValueError("At least one of 'min_score' and 'ratio' parameters "
                         "must be specified")

    return perm


def filter_adj(
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    perm: Tensor,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    According to a given node index is arranged in :obj:`perm` to
    filter and rearrange :obj:`edge_attr` and :obj:`edge_index`.

    Args:
        edge_index (Tensor): The representation of edges.
        edge_attr (Optional, Tensor): The attributes of the edges.
        perm (Tensor): Permutation tensor.
        num_nodes (Optinal, int): The number of nodes.
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class TopKPooling(torch.nn.Module):
    r"""
    Inherited from :class:`torch.nn.Module`, this class manages the top-k pooling;
    The top-k pooling operation selects the k largest values from the input tensor and discards the rest.

    Args:
        in_channels (int): The number of input channels.
        ratio (Union[float, int]): The selection ratio (default: :obj:`0.5`).
        min_score (Optional, float): Indicates the minimum score threshold. If :obj:`min_score` is specified,
        only nodes with scores greater than this threshold are selected (default: :obj:`None`).
        multiplier (float): A multiplication factor that adjusts the pooled node representation (default: :obj:`1`).
        nonlinearity (Union[str, Callable]): The attention scores of nodes are nonlinear transformed (default: :obj:`'tanh'`).
        softmax (bool): Whether to use softmax (default: :obj:`FALSE`).
        """

    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.,
        nonlinearity: Union[str, Callable] = 'tanh',
        softmax: bool = False,
    ):
        super().__init__()

        if isinstance(nonlinearity, str):
            nonlinearity = getattr(torch, nonlinearity)

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.softmax = softmax

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        uniform(self.in_channels, self.weight)

    def forward(
        self,
        x: Tensor,
        prompt: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:

        r"""Used to execute the module's forward computing logic."""

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = ((attn+prompt) * self.weight).sum(dim=-1)

        # pdb.set_trace()

        if (self.min_score is None) and (not self.softmax):
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)
        # pdb.set_trace()
        
        perm = topk(score, self.ratio, batch, self.min_score)
        # pdb.set_trace()
        x = score[perm].unsqueeze(1)*prompt

        # x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def __repr__(self) -> str:
        r"""return string representation"""
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')

class SAGPooling(torch.nn.Module):
    r"""
        Inherited from :class:`torch.nn.Module`;
        The most relevant or important nodes are selected according to their scores or attention coefficients,
        and pooled operations are performed to obtain the pooled node representation.

        Args:
            in_channels (int): The number of input channels.
            ratio (Union[float, int]): The selection ratio (default: :obj:`0.5`).
            GNN (torch.nn.Modile): The chosen GNN method (default: :obj:`GraphConv`).
            min_score (Optional, float): Indicates the minimum score threshold. If :obj:`min_score` is specified, only nodes with scores greater than this threshold are selected (default: :obj:`None`).
            multiplier (float): A multiplication factor that adjusts the pooled node representation (default: :obj:`1`).
            nonlinearity (Union[str, Callable]): The attention scores of nodes are nonlinear transformed (default: :obj:`'tanh'`).
            softmax (bool): Whether to use softmax (default: :obj:`FALSE`).
            **kwargs (dict): Additional attributes.
            """

    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        GNN: torch.nn.Module = GraphConv,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = 'tanh',
        softmax: bool = False,
        **kwargs,
    ):
        super().__init__()

        if isinstance(nonlinearity, str):
            nonlinearity = getattr(torch, nonlinearity)

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn = GNN(in_channels, 1, **kwargs)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity
        self.softmax = softmax

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.gnn.reset_parameters()

    def forward(
            self,
            x: Tensor,
            prompt: Tensor,
            edge_index: Tensor,
            edge_attr: OptTensor = None,
            batch: OptTensor = None,
            attn: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, OptTensor, Tensor, Tensor, Tensor]:
        r"""Used to execute the module's forward computing logic."""

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = self.gnn(attn+prompt, edge_index).view(-1)

        if (self.min_score is None) and (not self.softmax):
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)

        x = score[perm].unsqueeze(1)*prompt
        # x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def __repr__(self) -> str:
        r"""return the string representation."""
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.gnn.__class__.__name__}, '
                f'{self.in_channels}, {ratio}, multiplier={self.multiplier})')
