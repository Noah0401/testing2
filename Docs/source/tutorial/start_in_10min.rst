===============================
Quick Start
===============================


Basic Concepts
==============================

The core concept of `ProG` is to integrate the prompting engineering of graph neural network into a **single task**.

To get started with `ProG`, all you need to do is understand the following packages and their relationships, and then you'll be fluent in doing whatever you want with ProG.

.. contents::
    :local:

Tasker
-----------------------


A task is a list of parameters for the specific event you want to implement and give specific implementation.
A single task in `ProG` is described by an instance of class **prompt_graph.tasker.TaskType**, which holds the following attributes by default:

- TaskType: This indicate the type of task you want to perform (e.g. NodeTask, LinkTask, GraphTask).
- TaskType.pre_train_model_path: This indicates the path of your pre-trained model, the parameters of which will be read as the initial model.
- TaskType.prompt_type: This is prompting object of the prompt method you chose (e.g. All-in-one, GPPT, GPF, etc.).
- TaskType.gnn_type: This is the graph neural network object of the model you have chosen (e.g. GIN, GAT, GCN, etc.).
- TaskType.dataset_name: This is the dataset name string of the dataset you selected (e.g. Cora, CiteSeer, etc.).
- TaskType.run(): This function will automatically call the prompting and training process for us, and print the relevant metrics after training.

These properties are all built into the tasker object, and only need to be set in the external interface.

.. Note::
    As we mentioned before, integrating into tasks are the core building philosophy of ProG,
    we packaged all the details of graph cueing engineering into a task object, and by setting up the task object appropriately, you can achieve any kind of task
    on any model with any kind of dataset with any kind of pre-training method, **using** any kind of **prompting** method
    to achieve any kind of task at any level.

Let's show a concrete example of a design task:

.. code-block:: python

    from prompt_graph.tasker import NodeTask, LinkTask, GraphTask
    tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_Gprompt.GCN.pth', gnn_type='TransformerConv', hid_dim = 128, num_layer = 2,
                  dataset_name='Cora', prompt_type='GPF', epochs=100, shot_num=10, device : int = 5)
    tasker.run()


- pre_train_model_path: Used to set the ".pth" file path.
- gnn_type: Uesd to set the type of GNN model in your task.
- hid_dim: Used to set the dimension of hidden layers.
- num_layer: Used to set the layer of the GNN model of your task.
- dataset_name: Used to set the dataset you want to selet.
- prompt_type: Used to set the type of prompt you want to use in your task.
- epochs: Used to set the training times you want to perform.
- shot_num: Used to set the number of samples used in each training step.
- device: Used to set the device you want to use.


PreTrain
-------------------------


A pretrain object is used to build the pre-trained model on which the graph prompting project is based.
All you need to do is pretraining first and save the model under the relevant path. After that, you can reuse this pre-trained model.
A single pretain object in `ProG` is described by an instance of class **prompt_graph.pretain.preTainType**, which holds the following attributes by default:

- preTainType: This indicate the type of preTain you want to perform (e.g. DGI, GraphCL, SimGRACE, etc. ).
- preTainType.gnn_type: This is the graph neural network object of the model you have chosen (e.g. GIN, GAT, GCN, etc.), whose arguments will be saved.
- preTainType.dataset_name: This is the dataset name string of the dataset you selected (e.g. Cora, CiteSeer, etc.).
- preTainType.pretrain():  This function will automatically call the training process, print the relevant metrics after training and save the model to relevant file after training.

These properties are all built into the  preTrainType object, and only need to be set in the external interface.

.. Note::
    We have also packaged the pre-training process into an object. You can specify the model (e.g. GAT, GIN), dataset (e.g. Cora, CiteSeer),
    pre-training method (e.g. simGrace, EdgePred). After the pre-training, we will automatically save it to the "pre_trained_gnn" folder under your project file,
    named as "dataset_name+pre_train_method+GNN_model+node_per_layer" format. (e.g. "CiteSeer.Edgepred_GPPT.GCN.128hidden_dim.pth")

Let's show a concrete example of a design task:

.. code-block:: python

    from prompt_graph.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE
    from prompt_graph.utils import seed_everything
    from prompt_graph.utils import mkdir, get_args

    args = get_args()
    seed_everything(args.seed)
    mkdir('./pre_trained_gnn/')
    pt = SimGRACE(gnn_type = args.gnn_type, dataset_name = args.dataset_name, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)
    pt.pretrain()


- gnn_type: Uesd to set the type of GNN model in preTrain.
- dataset_name: Used to set the dataset you want to select in preTrain.
- hid_dim: Used to set the dim of the hidden layer of the GNN model in preTrain.
- gln: Used to set the layer of the GNN model in preTrain.
- num_epoch: Used to set the number of training epochs.
- device: Used to set the device you want to use.



Other Packages
------------

All other packages (data, evaluation, model, prompt, utils), are providing internal implementations to the task objects. If you just want to use `ProG` quickly, you don't need to know its internals.
Details can be seen in **Main Packages** part.


introduce with an example
==============================

For example, now we want to compare the node classification task without prompting and using the All-In-One prompting method.

Let's construct it step by step.

Firstly, let's overview the simple code.

.. code-block:: python

    from prompt_graph.tasker import NodeTask, GraphTask
    from prompt_graph.pretrain import Edgepred_GPPT, Edgepred_Gprompt, GraphCL, SimGRACE
    from prompt_graph.utils import seed_everything
    from torchsummary import summary
    from prompt_graph.utils import print_model_parameters
    from prompt_graph.utils import  mkdir, get_args

    # build a unified preTrained model
    args = get_args()
    seed_everything(args.seed)
    mkdir('./pre_trained_gnn/')
    pt = Edgepred_Gprompt(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs)
    pt.pretrain()
    # build different task with same pretrained model and run, compare them
    # tasker 1
    tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_Gprompt.GCN.pth',
                  dataset_name = args.dataset_name, num_layer = args.num_layer gnn_type = args.gnn_type, prompt_type = 'none', shot_num = 5)
    tasker.run()
    # tasker 2
    tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_Gprompt.GCN.pth',
                   dataset_name = args.dataset_name, num_layer = args.num_layer gnn_type = args.gnn_type, prompt_type = 'allinone', shot_num = 5)
    tasker.run()


Secondly, let's break it down bit by bit.

- Import relevant packages.

.. code-block:: python

    from prompt_graph.tasker import NodeTask
    from prompt_graph.pretrain import Edgepred_Gprompt
    from prompt_graph.utils import seed_everything
    from prompt_graph.utils import print_model_parameters
    from prompt_graph.utils import  mkdir, get_args


.. Note::
    You need to import the method you want to use for Pre-Train from
    **PreTrain** and import the level of the task you want to perform from **Tasker**

- PreTrain your model.

.. code-block:: python

    # build a unified preTrained model
    args = get_args()
    seed_everything(args.seed)
    mkdir('./pre_trained_gnn/')
    pt = Edgepred_Gprompt(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs)
    pt.pretrain()
    >>>


.. Note::
    Choose a pre-training parameter list and do a pre-train task, which you can generate randomly by seeding everything, or specify yourself.

- Compare two prompting methods.

.. code-block:: python

    # build different task with same pretrained model and run, compare them
    # tasker 1
    tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_Gprompt.GCN.pth',
                  dataset_name = args.dataset_name, num_layer = args.num_layer gnn_type = args.gnn_type, prompt_type = 'none', shot_num = 5)
    tasker.run()
    >>>
    # tasker 2
    tasker = NodeTask(pre_train_model_path = './pre_trained_gnn/Cora.Edgepred_Gprompt.GCN.pth',
                   dataset_name = args.dataset_name, num_layer = args.num_layer gnn_type = args.gnn_type, prompt_type = 'All-in-one', shot_num = 5)
    tasker.run()
    >>>


.. Note::
    Use a pre-trained model with a specified cue to do downstream and give an assessment of the effect.
    In this way, we can compare the accuracy, training complexity, etc. of different prompting methods.

Exercises
---------

1. What does "tasker.Tasktype" do?

2. Design a pre-training task and try to run it on your computer to see if it creates a ".pth" file locally.

3. Run script to see the difference between all the different prompting methods.