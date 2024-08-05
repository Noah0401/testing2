data
=======================

.. contents:: Contents
    :local:


batch
-------------------

.. currentmodule:: prompt_graph.data.batch

.. autosummary::
    :nosignatures:
    :template: autosummary/inherited_class.rst

    BatchFinetune
    BatchMasking
    BatchAE
    BatchSubstructContext

.. automodule:: prompt_graph.data.batch
    :members:
    :undoc-members:


dataloader
-------------------

.. currentmodule:: prompt_graph.data.dataloader

.. autosummary::
    :nosignatures:

    DataLoaderFinetune
    DataLoaderMasking
    DataLoaderAE
    DataLoaderSubstructContext

.. automodule:: prompt_graph.data.dataloader
    :members:
    :undoc-members:

graph_split
-------------------

.. currentmodule:: prompt_graph.data.graph_split

.. autosummary::
    :nosignatures:

    graph_split

.. automodule:: prompt_graph.data.graph_split
    :members:
    :undoc-members:

induced_graphs
-------------------

.. currentmodule:: prompt_graph.data.induced_graph

.. autosummary::
    :nosignatures:

    induced_graphs
    split_induced_graphs

.. automodule:: prompt_graph.data.induced_graph
    :members:
    :undoc-members:

load4data
-------------------

.. currentmodule:: prompt_graph.data.load4data

.. autosummary::
    :nosignatures:

    node_sample_and_save
    graph_sample_and_save
    load4graph
    load4node
    load4link_prediction_single_graph
    load4link_prediction_multi_graph
    NodePretrain

.. automodule:: prompt_graph.data.load4data
    :members:
    :undoc-members:


loader
-------------------

.. currentmodule:: prompt_graph.data.loader

.. autosummary::
    :nosignatures:

    nx_to_graph_data_obj
    graph_data_obj_to_nx
    BioDataset

.. automodule:: prompt_graph.data.loader
    :members:
    :undoc-members:



pooling
-------------------

.. currentmodule:: prompt_graph.data.pooling

.. autosummary::
    :nosignatures:

    topk
    filter_adj
    TopKPooling
    SAGPooling


.. automodule:: prompt_graph.data.pooling
    :members:
    :undoc-members:


