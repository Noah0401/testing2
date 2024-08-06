============
Introduction
============


Components
============

- ProG is composed of the following modules:
::

    - Data
    - Model
    - Pre-training
    - Prompt
    - Tasker
    - Evaluation
    - Utility



Explanation
============

- Our goal is to implement prompt's **tedious** code framework in *a few lines of command*, so as to facilitate the pipelines for prompting research.
- Graph Prompt Learning is a novel approach compared with traditional pre-training and fine-tuning methods.
- In the latter method, the pre-trained model requires modifications over time to adapt to different downstream tasks, which can be costly and troublesome.
- However, with GraphPrompt Learning, the pre-trained model can be "frozen", and the focus is on tuning the "prompt" for different tasks.
- ProG is used to support this idea. It has seven components: Data, Model, Pre-train, Prompt, Utilities, Tasker, and Evaluation

.. image:: https://github.com/sheldonresearch/ProG/blob/main/ProG_pipeline.jpg?raw=true
    :alt: pipline for ProG



Data
>>>>>>>>>

The Data module offers several methods for efficiently handling graph data or datasets:

1) Dataset processing: It provides methods for converting a dataset into batches, which allows
more efficient handling of large datasets. It also supports dataset loading with shuffling,
ensuring randomness in the order of data samples.

2) Dataset Splitting: It can divide a dataset into training, testing, and validation sets, which are
used for different purposes.

3) Inducing graphs: It provides methods to create induced graphs with predetermined sizes and
hops.

4) Graph Conversion: It allows to convert graph objects into the PyTorch Geometric (PyG)
"Data" objects and vice versa.

5) Graph Pruning: It can perform top-k pooling based on a given ”score” attribute. Top-k
pooling selects the top-k nodes from a graph based on a scoring mechanism, which is useful
for data processing tasks that require selecting the most relevant or important nodes.

Model
>>>>>>>>>

The Model part provides a set of Graph neural networks (GNNs) models for learning the representation of graph nodes and edges.

1) GAT (Graph Attention Network): It captures the relations of nodes by adaptively calcu-
lating the attention weights between nodes from stacking layers.

2) GCN (Graph Convolutional Network): Through scaling linearly in edges and learning from
hidden layers, it could capture the graph structure and nodes features.

3) GIN (Graph Isomorphism Network): It learns the representation of nodes by aggregating
its neighbors.

4) GraphSAGE (Graph Sample and Aggregated): Similar to GIN, it could learn node representations from neighbors and generate embedding for novel nodes by learning from previous
ones.

5) GraphTransformer: It is a graph representation learning model based on self-attention
mechanism, where both representations of nodes and edges can be learned.


Pretrain
>>>>>>>>>

The pre-train model offers a set of methods for unsupervised learning to initialize the model and
get the basic structure and feature information of graphs.

+-----------------+-------------------+
| Pre-train       | Task (N/E/G)      |
+=================+===================+
| DGI             | N                 |
+-----------------+-------------------+
| GraphMAE        | N                 |
+-----------------+-------------------+
| Edgepred_GPPT   | E                 |
+-----------------+-------------------+
| Edgepred_Gprompt| E                 |
+-----------------+-------------------+
| GraphCL         | G                 |
+-----------------+-------------------+
| SimGRACE        | G                 |
+-----------------+-------------------+

1) Node level: DGI and GraphMAE are two self-supervised learning methods for pre-trainning.
The former adopts the contrastive learning method to maximize mutual information between
patch representations and graphs, while the latter focuses on feature reconstruction.

2) Edge level: Both Edgepred-GPPT and Edgepred-GPrompt belong to it, which provides a
pre-train model for prompting and focus on edge level tasks.

3) Graph level: It involves GraphCL, a contrastive learning approach for acquiring graph representations, and SimGRACE, another contrastive learning method that considers both original
graphs and perturbed graphs as inputs.

Prompt
>>>>>>>>>

In this model, we have multiple prompts which can be classified into two different heuristics.

+-----------------+-------------------+
| Prompt          | (Prompt as) T/G   |
+=================+===================+
| All-in-one      | G                 |
+-----------------+-------------------+
| GPPT            | T                 |
+-----------------+-------------------+
| GPrompt         | T                 |
+-----------------+-------------------+
| GPF             | T                 |
+-----------------+-------------------+
| GPF-plus        | T                 |
+-----------------+-------------------+

1) Prompt as a token: This means treating a learnable vector p as a prompt, which can be
viewed as an adding feature to the original feature. GPPT, GPrompt, GPF, and GPF-plus
mainly used this method.

2) Prompt as a graph: All-in-one belongs to this, which designs a prompt graph G = (P, S),
where P contains a set of tokens.

Tasker
>>>>>>>>>

This module provides specific downstream task implementation for node classification, link prediction, and graph classification.

1) Node task: This task involves creating data folders, loading data or datasets, loading induced
graphs, training GNN and multiple prompt models, and finally performing training to obtain
the accuracy level of node classification.

2) Edge task: The edge task involves loading data or datasets, initializing the GNN model, loss
function, and optimizer, training the link prediction model using a negative sampling strategy,
and testing while returning ROC and AUC scores.

3) Graph task: Similar to the previous tasks, the graph task involves initialization, data loading, prompt training, and multiple steps while implementing graph classification tasks, and
returning the accuracy level.


Evaluation
>>>>>>>>>>>

This module is used for evaluating the performance of different prompt models, which calculates
node or graph classification accuracy.

Utils
>>>>>>>>>>>

This module provides sets of functions that can be used to facilitate later taskers.

1) Data pre-processing: It involves a series of actions such as converting an edge index into a
sparse matrix, calculating the average embedding vector for each class in the input feature,
calculating the distance between each sample in the input feature and the class center
embedding vector, perturbing data by dropping nodes, masking nodes or perturbing
edges and so on.

2) Model pre-processing: It involves various activation functions, loss functions,
and constraints.

3) Auxiliary functions: It also provides functions to print data and models, creat folders, and
seed everything.


Datasets
============

- we also summarized various kind of datasets in prompt research:

+-----------+---------------+------------+------------+---------------+--------------+------------+
| Graphs    | Graph classes | Avg. nodes | Avg. edges | Node features | Node classes | Task       |
+===========+===============+============+============+===============+==============+============+
| Cora      | 1             | 2,708      | 5,429      | 1,433         | 7            | N          |
+-----------+---------------+------------+------------+---------------+--------------+------------+
| Pubmed    | 1             | 19,717     | 88,648     | 500           | 3            | N          |
+-----------+---------------+------------+------------+---------------+--------------+------------+
| CiteSeer  | 1             | 3,327      | 9,104      | 3,703         | 6            | N          |
+-----------+---------------+------------+------------+---------------+--------------+------------+
| Mutag     | 188           | 17.9       | 39.6       | ?             | 7            | N          |
+-----------+---------------+------------+------------+---------------+--------------+------------+
| Reddit    | 1             | 232,965    | 23,213,838 | 602           | 41           | N          |
+-----------+---------------+------------+------------+---------------+--------------+------------+
| Amazon    | 1             | 13,752     | 491,722    | 767           | 10           | N          |
+-----------+---------------+------------+------------+---------------+--------------+------------+
| Flickr    | 1             | 89,250     | 899,756    | 500           | 7            | N          |
+-----------+---------------+------------+------------+---------------+--------------+------------+
| PROTEINS  | 1,113         | 39.06      | 72.82      | 1             | 3            | N, G       |
+-----------+---------------+------------+------------+---------------+--------------+------------+
| ENZYMES   | 600           | 32.63      | 62.14      | 18            | 3            | N, G       |
+-----------+---------------+------------+------------+---------------+--------------+------------+


