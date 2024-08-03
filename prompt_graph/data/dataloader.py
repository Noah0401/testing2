import torch.utils.data
from torch.utils.data.dataloader import default_collate

from .batch import BatchFinetune, BatchMasking, BatchAE, BatchSubstructContext

class DataLoaderFinetune(torch.utils.data.DataLoader):
    r"""Merges data objects from a
        :class:`torch_geometric.data.Dataset` to a mini-batch for fine-tune mission.
        :class:`torch.utils.data.DataLoader` could support for map-style and iterable-style datasets,
        customizing data loading order, automatic batching, single and multi-process data loading,
        and automatic memory pinning.

        Args:
            dataset (Dataset): The dataset from which to load the data.
            batch_size (int, optional): How may samples per batch to load.
                (default: :obj:`1`).
            shuffle (bool, optional): If set to :obj:`True`, the data will be
                reshuffled at every epoch (default: :obj:`True`).
            **kwargs (dict): Additional attributes for :class:`torch.utils.data.DataLoader`.
        """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderFinetune, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchFinetune.from_data_list(data_list),
            **kwargs)

class DataLoaderMasking(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
        :class:`torch_geometric.data.dataset` to a mini-batch for masking mission.

        Args:
            dataset (Dataset): The dataset from which to load the data.
            batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`).
            shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`).
            **kwargs (dict): Additional attributes for :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)


class DataLoaderAE(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch for autoencoder mission.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`).
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`).
        **kwargs (dict): Additional attributes for :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAE.from_data_list(data_list),
            **kwargs)


class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
        :class:`torch_geometric.data.dataset` to a mini-batch for
        those with substructure context pair.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`).
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`).
        **kwargs (dict): Additional attributes for :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list),
            **kwargs)



