import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np


class downprompt(nn.Module):
    r"""
        Inherit from :class:`torch.nn.Module`, it defines a downprompt model,
        which combines multiple prompt.

        Args:
            prompt1, prompt2, prompt3 (Tensor): The number of input channels.
            a4 (Any)
            ft_in (int): The input feature.
            nb_classes (int): Number of classes.
            device (torch.device): The type of the device.
        """

    def __init__(self, prompt1, prompt2, prompt3, a4, ft_in, nb_classes, device):
        super(downprompt, self).__init__()
        self.downprompt = downstreamprompt(ft_in)
        self.nb_classes = nb_classes
        self.a4 = a4
        self.leakyrelu = nn.ELU()
        self.device = device
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        self.nodelabelprompt = weighted_prompt(3)
        self.dffprompt = weighted_feature(2)
        self.aveemb0 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb1 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb2 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb3 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb4 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb5 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb6 = torch.FloatTensor(ft_in, ).to(self.device)

        self.one = torch.ones(1, ft_in).to(self.device)
        self.ave = torch.FloatTensor(nb_classes, ft_in).to(self.device)

    def forward(self, seq, seq1, labels, train=0):

        weight = self.leakyrelu(self.nodelabelprompt(self.prompt))
        weight = self.one + weight
        rawret1 = weight * seq
        rawret2 = self.downprompt(seq)
        rawret4 = seq1
        # rawret3 = rawret1 + rawret2
        rawret3 = self.dffprompt(rawret1, rawret2)
        # # print("a4",self.a4,"a5",self.a5)

        rawret = rawret3 + self.a4 * rawret4

        # rawret = seq
        rawret = rawret.to(self.device)
        # rawret = torch.stack((rawret,rawret,rawret,rawret,rawret,rawret))
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret, nb_class=self.nb_classes).to(self.device)

        ret = torch.FloatTensor(seq.shape[0], self.nb_classes).to(self.device)
        # print("avesize",self.ave.size(),"ave",self.ave)
        # print("rawret=", rawret[1])
        # print("aveemb", self.ave)
        for x in range(0, seq.shape[0]):
            ret[x][0] = torch.cosine_similarity(rawret[x], self.ave[0], dim=0)
            ret[x][1] = torch.cosine_similarity(rawret[x], self.ave[1], dim=0)
            ret[x][2] = torch.cosine_similarity(rawret[x], self.ave[2], dim=0)
            ret[x][3] = torch.cosine_similarity(rawret[x], self.ave[3], dim=0)
            ret[x][4] = torch.cosine_similarity(rawret[x], self.ave[4], dim=0)
            ret[x][5] = torch.cosine_similarity(rawret[x], self.ave[5], dim=0)
            if self.nb_classes == 7:
                ret[x][6] = torch.cosine_similarity(rawret[x], self.ave[6], dim=0)

        ret = F.softmax(ret, dim=1)

        # ret = torch.argmax(ret, dim=1)
        # print('ret=', ret)

        return ret

    def weights_init(self, m):
        r"""Initialize the weight of the mode."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


def averageemb(labels, rawret, nb_class) -> torch.Tensor:
    r"""
    Calculates the average embedding vector for each class.

    Args:
        labels (Tensor): Contains the information of labels.
        rawret (Tensor): The original data.
        nb_class (int): The number of classes.
    """
    retlabel = torch.FloatTensor(nb_class, int(rawret.shape[0] / nb_class), int(rawret.shape[1]))
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    cnt7 = 0
    # print("labels",labels)
    for x in range(0, rawret.shape[0]):
        if labels[x].item() == 0:
            retlabel[0][cnt1] = rawret[x]
            cnt1 = cnt1 + 1
        if labels[x].item() == 1:
            retlabel[1][cnt2] = rawret[x]
            cnt2 = cnt2 + 1
        if labels[x].item() == 2:
            retlabel[2][cnt3] = rawret[x]
            cnt3 = cnt3 + 1
        if labels[x].item() == 3:
            retlabel[3][cnt4] = rawret[x]
            cnt4 = cnt4 + 1
        if labels[x].item() == 4:
            retlabel[4][cnt5] = rawret[x]
            cnt5 = cnt5 + 1
        if labels[x].item() == 5:
            retlabel[5][cnt6] = rawret[x]
            cnt6 = cnt6 + 1
        if labels[x].item() == 6:
            retlabel[6][cnt7] = rawret[x]
            cnt7 = cnt7 + 1
    retlabel = torch.mean(retlabel, dim=1)
    return retlabel


class weighted_prompt(nn.Module):
    r"""
    Inherited from :class:`torch.nn.Module`, it defines a weighted prompt model;
    The input graph is being weighted.

    Args:
        weightednum (int): The number of weight.
    """

    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)
        r"""
        Fills the different positions of :obj:`self.weight` with the specified values:
        :obj:`self.weight[0][0]` is :obj:`0.5`,
        :obj:`self.weight[0][1]` is :obj:`0.4` and
        :obj:`self.weight[0][2]` is :obj:`0.3`.
        """
        self.weight[0][0].data.fill_(0.5)
        self.weight[0][1].data.fill_(0.4)
        self.weight[0][2].data.fill_(0.3)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding = torch.mm(self.weight, graph_embedding)
        return graph_embedding


class weighted_feature(nn.Module):
    r"""
    Inherit fromed :class:`torch.nn.Module`;
    The graph embeddings of the two inputs are weighted and the activation function is applied.

    Args:
        weightednum (int): The number of weight.
    """

    def __init__(self, weightednum):
        super(weighted_feature, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)
        r"""
            Fills the different positions of :obj:`self.weight` with the specified values:
            :obj:`self.weight[0][0]` is :obj:`0`,
            :obj:`self.weight[0][1]` is :obj:`1`.
                """
        self.weight[0][0].data.fill_(0)
        self.weight[0][1].data.fill_(1)

    def forward(self, graph_embedding1, graph_embedding2):
        # print("weight",self.weight)
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)


class downstreamprompt(nn.Module):
    r"""
    Inherited from :class:`torch.nn.Module`;
    The graph embedding is weighted by weight parameters,
    and the importance of different features can be controlled by adjusting the value of weight parameters,
    thus affecting the output of the model.

    Args:
        hid_units (int): The number of hidden units.
    """

    def __init__(self, hid_units):
        super(downstreamprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        r"""Initializes the parameters."""
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(0.3)
        # self.weight[0][1].data.fill_(0.3)
        # self.weight[0][2].data.fill_(0.3)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding = self.weight * graph_embedding
        return graph_embedding


class featureprompt(nn.Module):
    r"""
        Inherited from :class:`torch.nn.Module`;
        The input features are weighted.

        Args:
            prompt1, prompt2, prompt3 (Tensor): 3 different prompts.
        """

    def __init__(self, prompt1, prompt2, prompt3):
        super(featureprompt, self).__init__()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        self.weightprompt = weighted_prompt(3)

    def forward(self, feature):
        # print("prompt",self.weightprompt.weight)
        weight = self.weightprompt(self.prompt)
        feature = weight * feature
        return feature


class GCN(nn.Module):
    r"""
        Inherited from :class:`torch.nn.Module`;
        Defines a GCN model, which contains a linear mapping layer and an activation function;
        More information about GCN can be found in the previous module.
            """

    def __init__(self, in_ft, out_ft, act=None, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        r"""
        Initializes the weight.

        Args:
            m (object): Check whether module m is a linear layer.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, input, sparse=True):
        # print("input",input)
        seq = input[0]
        adj = input[1]
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.spmm(adj, seq_fts)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class GcnLayers(torch.nn.Module):
    r"""
    Inherited from :class:`torch.nn.Module`;
    Forward propagation of convolutional network layers containing multiple graphs is realized.

    Args:
        n_in (int): The dimension of input.
        n_h (int): The dimension of hidden layers.
        num_layers_num (int): The number of GCN layers.
        dropout (float): Dropout rate.
    """

    def __init__(self, n_in, n_h, num_layers_num, dropout):
        super(GcnLayers, self).__init__()

        self.act = torch.nn.ReLU()
        self.num_layers_num = num_layers_num
        self.g_net, self.bns = self.create_net(n_in, n_h, self.num_layers_num)

        self.dropout = torch.nn.Dropout(p=dropout)

    def create_net(self, input_dim, hidden_dim, num_layers):
        r"""
        Creates graph convolution layer and batch normalization layer.
        """
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                nn = GCN(hidden_dim, hidden_dim)
            else:
                nn = GCN(input_dim, hidden_dim)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns

    def forward(self, seq, adj, sparse, LP=False):
        graph_output = torch.squeeze(seq, dim=0)
        graph_len = adj
        xs = []
        for i in range(self.num_layers_num):
            input = (graph_output, adj)
            graph_output = self.convs[i](input)
            if LP:
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
            xs.append(graph_output)

        return graph_output.unsqueeze(dim=0)


class Discriminator(nn.Module):
    r"""
       Inherited from :class:`torch.nn.Module`,
       discriminating generate graphs and real graphs.

       Args:
           n_h (int): The dimension of hidden layers.
       """

    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        r"""Initializes the weight.

            Args:
                m (object): Check whether module m is a linear layer."""
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    r"""
        Inherited from :class:`torch.nn.Module`;
        Applies an average on seq, of shape (:obj:`batch, nodes, features`)
        While taking into account the masking of :obj:`msk`.
        """

    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class DGI(nn.Module):
    r"""
    Inherited from :class:`torch.nn.Module`, creating DGI model;
    The graph convolution operation is followed by feature multiplication.

    Args:
        n_h: The dimension of hidden features.
    """

    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = gcn(seq1, adj, sparse)

        # print("h_1",h_1.shape)

        h_3 = h_1 * self.prompt

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = gcn(seq2, adj, sparse)

        h_4 = h_2 * self.prompt

        ret = self.disc(c, h_3, h_4
                        , samp_bias1, samp_bias2)

        return ret

    def reset_parameters(self):
        r"""Resets the parameters."""
        torch.nn.init.xavier_uniform_(self.prompt)


class DGIprompt(nn.Module):
    r"""
    Inherited from :class:`torch.nn.Module`, creating DGIprompt model;
    The graph convolution operation is done before feature multiplication.

    Args:
        n_in: The dimension of input features.
        n_h: The dimension of hidden features.
    """

    def __init__(self, n_in, n_h, activation):
        super(DGIprompt, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.prompt = nn.Parameter(torch.FloatTensor(1, n_in), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        seq1 = seq1 * self.prompt
        h_1 = gcn(seq1, adj, sparse)

        # print("h_1",h_1.shape)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        seq2 = seq2 * self.prompt
        h_2 = gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2
                        , samp_bias1, samp_bias2)

        return ret

    def reset_parameters(self):
        r"""Resets the parameters."""
        torch.nn.init.xavier_uniform_(self.prompt)


class GraphCL(nn.Module):
    r"""
        Inherited from :class:`torch.nn.Module`, creating GraphCL model.

        Args:
            n_h: The dimension of hidden features.
        """

    def __init__(self, n_in, n_h, activation):
        super(GraphCL, self).__init__()
        #  self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2,
                aug_type):

        h_0 = gcn(seq1, adj, sparse)

        h_00 = h_0 * self.prompt
        if aug_type == 'edge':

            h_1 = gcn(seq1, aug_adj1, sparse)
            h_3 = gcn(seq1, aug_adj2, sparse)

        elif aug_type == 'mask':

            h_1 = gcn(seq3, adj, sparse)
            h_3 = gcn(seq4, adj, sparse)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = gcn(seq3, aug_adj1, sparse)
            h_3 = gcn(seq4, aug_adj2, sparse)

        else:
            assert False

        h_11 = h_1 * self.prompt
        h_33 = h_3 * self.prompt

        c_1 = self.read(h_11, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_33, msk)
        c_3 = self.sigm(c_3)

        h_2 = gcn(seq2, adj, sparse)

        h_22 = h_2 * self.prompt

        ret1 = self.disc(c_1, h_00, h_22, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_00, h_22, samp_bias1, samp_bias2)

        ret = ret1 + ret2
        return ret

    def reset_parameters(self):
        r"""
        Resets the parameters.
        """
        torch.nn.init.xavier_uniform_(self.prompt)


class GraphCLprompt(nn.Module):
    r"""
        Inherited from :class:`torch.nn.Module`, creating GraphCLprompt model.

        Args:
            n_in: The dimension of input features.
            n_h: The dimension of hidden features.
        """

    def __init__(self, n_in, n_h, activation):
        super(GraphCLprompt, self).__init__()
        #  self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_in), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2,
                aug_type):

        seq1 = seq1 * self.prompt
        seq2 = seq2 * self.prompt
        seq3 = seq3 * self.prompt
        seq4 = seq4 * self.prompt

        h_0 = gcn(seq1, adj, sparse)

        if aug_type == 'edge':

            h_1 = gcn(seq1, aug_adj1, sparse)
            h_3 = gcn(seq1, aug_adj2, sparse)

        elif aug_type == 'mask':

            h_1 = gcn(seq3, adj, sparse)
            h_3 = gcn(seq4, adj, sparse)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = gcn(seq3, aug_adj1, sparse)
            h_3 = gcn(seq4, aug_adj2, sparse)

        else:
            assert False

        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_3, msk)
        c_3 = self.sigm(c_3)

        h_2 = gcn(seq2, adj, sparse)

        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)
        ret = ret1 + ret2
        return ret

    def reset_parameters(self):
        r"""
        Resets the parameters.
        """
        torch.nn.init.xavier_uniform_(self.prompt)


class LogReg(nn.Module):
    r"""
        Inherited from :class:`torch.nn.Module`,
        creating a logistic regression model for mapping input features to categories of classification.

        Args:
            ft_in: The number of input features.
            nb_classes: The number of classes.
        """

    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """
        Initialize the weight;
        Use the Xavier uniformly distributed initialization method (:obj:`Torch.nn.inin.xavier_Uniform_`)
        to initialize the weights and set the bias to :obj:`0`.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Lp(nn.Module):
    r"""
        Inherited from :class:`torch.nn.Module`,
        The input sequence and adjacency matrix are generated by graph convolution operation
        and parameter multiplication,
        and then activated by activation function.

        Args:
            n_h (int): The dimension of hidden features.
        """

    def __init__(self, n_in, n_h):
        super(Lp, self).__init__()
        self.sigm = nn.ELU()
        self.act = torch.nn.LeakyReLU()
        # self.dropout=torch.nn.Dropout(p=config["dropout"])
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq, adj, sparse):
        h_1 = gcn(seq, adj, sparse, True)
        # ret = h_1
        ret = h_1 * self.prompt
        # ret = h_1
        # print("ret1",ret)
        ret = self.sigm(ret.squeeze(dim=0))
        # print("ret2",ret)
        # ret = ret.squeeze(dim=0)
        return ret

    def reset_parameters(self):
        r"""
        Resets the parameters.
        """
        torch.nn.init.xavier_uniform_(self.prompt)


class Lpprompt(nn.Module):
    r"""
        Inherited from :class:`torch.nn.Module`,
        accepts input sequences and adjacency matrices and generates hidden layer features
        through parametric multiplication and graph convolution operations.

        Args:
            n_in (int): The dimension of the input.
        """

    def __init__(self, n_in, n_h):
        super(Lpprompt, self).__init__()
        self.sigm = nn.ELU()
        self.act = torch.nn.LeakyReLU()
        # self.dropout=torch.nn.Dropout(p=config["dropout"])
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_in), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq, adj, sparse):
        seq = seq * self.prompt
        h_1 = gcn(seq, adj, sparse, True)
        ret = h_1
        # ret = h_1 * self.prompt
        # ret = h_1
        # print("ret1",ret)
        ret = self.sigm(ret.squeeze(dim=0))
        # print("ret2",ret)
        # ret = ret.squeeze(dim=0)
        return ret

    def reset_parameters(self):
        r"""Resets the parameters."""
        torch.nn.init.xavier_uniform_(self.prompt)