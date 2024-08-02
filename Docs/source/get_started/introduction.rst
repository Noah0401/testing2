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

- Our goal is to implement prompt's **tedious** code framework in *a few lines of command*, so as to facilitate the pipelines for prompting research

.. image:: https://github.com/sheldonresearch/ProG/blob/main/ProG_pipeline.jpg?raw=true
    :alt: pipline for ProG

Data
>>>>>>>>>

Provides basic methods for working with graph data or data sets

Model
>>>>>>>>>

Provides several models such as **GAT**, **GCN** for supervised training.

Pretrain
>>>>>>>>>

Provides several pre-train methods which covers **node level**, **edge level**, and **graph level** pre-training.

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

Prompt
>>>>>>>>>

Provides several prompt methods. The generated prompts can be classified as **token** and **graph**.

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

Tasker
>>>>>>>>>

Provides node, edge and graph task implementation.

Evaluation
>>>>>>>>>>>

Provides evaluation methods for different tasks corresponding to different Prompts.

Utils
>>>>>>>>>>>

Provides basic utilities of the methods which can deal with repetitive tasks.



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


