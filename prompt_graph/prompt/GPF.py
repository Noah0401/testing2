import torch
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F


class GPF(torch.nn.Module):
    r"""Inherit from :class:`torch.nn.Module`;
    GPF can make adjustments to the features of the input graph to introduce additional cues
    or to influence the features globally;
    See `here <https://arxiv.org/pdf/2209.15240>`__ for more information.

    Argus:
        in_channels (int): The number of channels for the input tensor.
    """
    def __init__(self, in_channels: int):
        super(GPF, self).__init__()
        self.global_emb = torch.nn.Parameter(torch.Tensor(1,in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        r"""Initializes :obj:`self.global_emb`."""
        glorot(self.global_emb)

    def add(self, x: torch.Tensor):
        r"""Adds up :obj:`self.global_emb` and :obj:`x`."""
        return x + self.global_emb

class GPF_plus(torch.nn.Module):
    r"""Inherited from :class:`torch.nn.Module`;
        GPF-plus uses different prompt features on different nodes in the graph;
        See `here <https://arxiv.org/pdf/2209.15240>`__ for more information.

        Argus:
            in_channels (int): The number of channels for the input tensor.
            p_num (int): The number of prompts.
        """
    def __init__(self, in_channels: int, p_num: int):
        super(GPF_plus, self).__init__()
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Initializes the parameters: :obj:`self.p_list` and :obj:`self.a`.
        """
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: torch.Tensor):
        r"""
        Adds up :obj:`x` and :obj:`self.p_list`
        """
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p

