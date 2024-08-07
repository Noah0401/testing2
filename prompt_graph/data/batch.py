import torch
from torch_geometric.data import Data, Batch

class BatchFinetune(Data):
    r"""Inherited from :class:`torch_geometric.data.Data`,
    it can create a :obj:`batch` for fine-tuning; A :obj:`batch` is a combination of several graphs in a dataset, and is
    represented by a tensor which maps each node to its respective graph;
    :obj:`Fine-tune` is a process after pre-train, and completing fine-tuning process with the help of
    batches could enhance the efficiency.

    Args:
        batch (torch.Tensor): The created batch tensor.
        **kwargs (dict): Additional attributes of :class:`torch_geometric.data.Data`.
        """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list: list[any])->torch.Tensor:
        r"""Constructs a :obj:`batch` object from a python list holding
            :class:`torch_geometric.data.Data` objects.
            The assignment vector :obj:`batch` is created on the fly.

            Args:
                data_list (list[any]): The objects to be created into a batch.
            """

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'center_node_idx']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].cat_dim(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self)->int:
        r"""Returns the number of graphs in a batch."""
        return self.batch[-1].item() + 1


class BatchMasking(Data):
    r"""Inherited from :class:`torch_geometric.data.Data`, it can create a
        :obj:`batch` for masking; And :obj:`masking` for both nodes and edges means hiding partial feature information, which
        controls a model's access to and processing of different elements in the graph.

        Args:
            batch (torch.Tensor): The created batch matrix.
            **kwargs (dict): Additional attributes of :class:`torch_geometric.data.Data`.
        """
    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list: list[any])->torch.Tensor:
        r"""Constructs a batch object from a python list holding
            :class:`torch_geometric.data.Data` objects.
            The assignment tensor :obj:`batch` is created on the fly.

            Args:
                data_list (list[any]): The objects to be created into a batch.
            """
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index']:
                    item = item + cumsum_node
                elif key  == 'masked_edge_idx':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].cat_dim(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key:str, item)->bool:

        r"""This is a prompt function, If it returns :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.

         .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.

        Args:
            key (str): Keywords used for judgement.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self)->int:
        r"""Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchAE(Data):
    r"""Inherited from :class:`torch_geometric.data.Data`, it can create a
        :obj:`batch` for data with different autoencoders, :obj:`AE` stands for "Autoencoder";
        Autoencoders are unsupervised learning algorithms used to learn efficient data
        representations, typically for dimensionality reduction or feature learning; While :obj:`BatchAE` is a graph embedding method based on autoencoder.

        Args:
            batch (torch.Tensor): The created batch matrix.
            **kwargs (dict): Additional attributes of :class:`torch_geometric.data.Data`.
        """

    def __init__(self, batch=None, **kwargs):
        super(BatchAE, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list: list[any])->torch.Tensor:
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects. The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchAE()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'negative_edge_index']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self) ->int:
        r"""Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def cat_dim(self, key:str) ->int:
        r"""Determines the concatenation dimension when using the :obj:`torch.cat`
            function for tensor concatenation. """

        return -1 if key in ["edge_index", "negative_edge_index"] else 0



class BatchSubstructContext(Data):
    r""" Inherited from :class:`torch_geometric.data.Data`, it creates
    specialized batch for substructure context pair.

    Args:
        batch (torch.Tensor): The created batch matrix.
        **kwargs (dict): Additional attributes of :class:`torch_geometric.data.Data`.
        """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list: list[any])->torch.Tensor:
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        #keys = [set(data.keys) for data in data_list]
        #keys = list(set.union(*keys))
        #assert 'batch' not in keys

        batch = BatchSubstructContext()
        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]

        for key in keys:
            #print(key)
            batch[key] = []

        #batch.batch = []
        #used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0

        for data in data_list:
            #If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                #batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ###batching for the main graph
                #for key in data.keys:
                #    if not "context" in key and not "substruct" in key:
                #        item = data[key]
                #        item = item + cumsum_main if batch.cumsum(key, item) else item
                #        batch[key].append(item)

                ###batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item)


                ###batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.cat_dim(key))
        #batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        return batch.contiguous()

    def cat_dim(self, key:str)->int:
        r"""Determines the concatenation dimension when using the :obj:`torch.cat`
        function for tensor concatenation"""
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key:str, item)->bool:
        r"""If the function returns :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.

        .. note::
            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]

    @property
    def num_graphs(self)->int:
        r"""Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
