import torch

class Gprompt(torch.nn.Module):
    r"""
    Inherit from :class:`torch.nn.Module`, it defines a GPrompt model, which hinges on a learnable
    prompt to actively guide downstream tasks using task-specific
    aggregation in :obj:`ReadOut`, in order to drive the downstream tasks
    to exploit the pre-trained model in a task-specific manner;
    See `here <https://arxiv.org/abs/2302.08043>`__ for more information.

    Args:
        input_dim (int): The dimension of the input.
    """
    def __init__(self,input_dim):
        super(Gprompt, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.max_n_num=input_dim
        self.reset_parameters()
    def reset_parameters(self):
        r"""
        Initializes the parameter :obj:`self.weight.`
        """
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, node_embeddings):
        node_embeddings=node_embeddings*self.weight
        # graph_embedding=graph_embedding.sum(dim=1) 有一点点抽象
        return node_embeddings