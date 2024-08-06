import torch
from sklearn.cluster import KMeans
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class SimpleMeanConv(MessagePassing):
    r"""
    Inherit from :class:`torch_geometric.nn.MessagePassing`,
    complete GCN with :class:`mean` being its aggregation mode.
    """

    def __init__(self):
        # 初始化时指定聚合方式为 'mean'，即平均聚合
        super(SimpleMeanConv, self).__init__(aggr='mean')  # 'mean'聚合。

    def forward(self, x, edge_index):
        # x 代表节点特征矩阵，edge_index 是图的边索引列表

        # 在边索引中添加自环，这样在聚合时，节点也会考虑自己的特征
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 开始消息传递过程，其中x是每个节点的特征，edge_index定义了节点间的连接关系
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        r"""Returns the feature of neighbor node.

        Args:
            x_j (Tensor): Features of the neighbor nodes.
        """
        # x_j 表示邻居节点的特征，这里直接返回，因为我们使用的是 'mean' 聚合
        return x_j


class GPPTPrompt(torch.nn.Module):
    r"""
    Inherited from :class:`torch.nn.Module`, it defines a GPPTPrompt model;
    GPPT converts the node to a token pair;
    A Token pair contains a task tag (to represent the node label) and a structure tag (to describe the node);
    See `here <https://dl.acm.org/doi/pdf/10.1145/3534678.3539249>`__ for more information.

    Args:
        n_hidden (int): The number of hidden layers.
        center_num (int): The number of central nodes.
        n_classes (int): The number of classes.
        device (torch.device): The type of the device.
    """

    def __init__(self, n_hidden, center_num, n_classes, device):
        super(GPPTPrompt, self).__init__()
        self.center_num = center_num
        self.n_classes = n_classes
        self.device = device
        self.StructureToken = torch.nn.Linear(n_hidden, center_num, bias=False)
        self.StructureToken = self.StructureToken.to(device)  # structure token
        self.TaskToken = torch.nn.ModuleList()
        for i in range(center_num):
            self.TaskToken.append(torch.nn.Linear(2 * n_hidden, n_classes, bias=False))  # task token
        self.TaskToken = self.TaskToken.to(device)

    def weigth_init(self, h, edge_index, label, index):
        r"""Initialize the weights of structure tokens
         by aggregating operation and K-Means algorithm.

         Args:
             h (Tensor): The feature of a node in the graph.
             edge_index (Tensor): The index of edges.
             label (Tensor): The label of the graph.
             index (Tensor): Used to select a specific node from node feature.

         """
        # 对于图中的每一个节点，将其特征（'h'）发送给所有邻居节点，然后每个节点会计算所有收到的邻居特征的平均值，并将这个平均值存储为自己的新特征在'neighbor'下

        conv = SimpleMeanConv()
        # 使用这个层进行前向传播，得到聚合后的节点特征
        h = conv(h, edge_index)

        features = h[index]
        labels = label[index.long()]
        cluster = KMeans(n_clusters=self.center_num, random_state=0).fit(features.detach().cpu())

        temp = torch.FloatTensor(cluster.cluster_centers_).to(self.device)
        self.StructureToken.weight.data = temp.clone().detach()

        p = []
        for i in range(self.n_classes):
            p.append(features[labels == i].mean(dim=0).view(1, -1))
        temp = torch.cat(p, dim=0).to(self.device)
        for i in range(self.center_num):
            self.TaskToken[i].weight.data = temp.clone().detach()

    def update_StructureToken_weight(self, h):
        r"""
        Update the weights of structure tokens by K-Means algorithm.

        Args:
             h (Tensor): The feature of a node in the graph.
        """
        cluster = KMeans(n_clusters=self.center_num, random_state=0).fit(h.detach().cpu())
        temp = torch.FloatTensor(cluster.cluster_centers_).to(self.device)
        self.StructureToken.weight.data = temp.clone().detach()

    def get_TaskToken(self):
        r"""Returns the task tokens."""
        pros = []
        for name, param in self.named_parameters():
            if name.startswith('TaskToken.'):
                pros.append(param)
        return pros

    def get_StructureToken(self):
        r"""Returns the task tokens."""
        for name, param in self.named_parameters():
            if name.startswith('StructureToken.weight'):
                pro = param
        return pro

    def get_mid_h(self):
        r"""Returns the node feature."""
        return self.fea

    def forward(self, h, edge_index):
        device = h.device
        conv = SimpleMeanConv()
        # 使用这个层进行前向传播，得到聚合后的节点特征
        h = conv(h, edge_index)
        self.fea = h

        out = self.StructureToken(h)
        index = torch.argmax(out, dim=1)
        out = torch.FloatTensor(h.shape[0], self.n_classes).to(device)
        for i in range(self.center_num):
            out[index == i] = self.TaskToken[i](h[index == i])
        return out

