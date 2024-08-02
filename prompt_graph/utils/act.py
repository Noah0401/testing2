import torch
import torch.nn.functional as F


def act(x=None, act_type:str='leakyrelu'):
    r"""
    Applies different activation functions to the input x based on the specified act_type.

    Args:
        x (Tensor): The input tensor. (default: :obj:`None`)
        act_type (str): The specified activation function type. (default: :obj:`leakyrelu`)
    """
    if act_type == 'leakyrelu':
        return torch.nn.LeakyReLU() if x is None else F.leaky_relu(x)
    elif act_type == 'tanh':
        return torch.nn.Tanh() if x is None else torch.tanh(x)
    elif act_type == 'relu':
        return torch.nn.ReLU() if x is None else F.relu(x)
    elif act_type == 'sigmoid':
        return torch.nn.Sigmoid() if x is None else torch.sigmoid(x)
    elif act_type == 'softmax':
        # 注意：softmax 需要指定维度；这里假设对最后一个维度进行softmax
        return torch.nn.Softmax(dim=-1) if x is None else F.softmax(x, dim=-1)
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")

# exmaple
# act_type = 'relu', 'sigmoid', 'softmax', 'tanh', 'leakyrelu'
# x = torch.tensor(...)
# result = act(x, act_type='relu')
