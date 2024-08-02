Pre-Train
============

- We summarized all the possible ways of pre-training in **academic research**.
Including:
::

    - Edge Prediction
    - GraphCL
    - SimGRACE
    - and even more


- To pre-train your model you basically need the following steps.

    + **Firstly**: Determine the model, the hidden dimension and the hidden layers you want to set.
    .. code-block:: python

        gln = number of hidden layers
        hid_dim = hidden dimension
        gnn_type = model you what use

    + **Secondly**: Determine the dataset and how many shots you what set.
    .. code-block:: python

        dataname = dataset you want to use
        num_parts =  shots you what to use
        graph_list, input_dim, hid_dim = load_data4pretrain(dataname, num_parts)

    + **Thirdly**: Determine the pre-train method you want to use and build the task of pretrain.
    .. code-block:: python

        pt = PreTrain(pre_train_method, gnn_type, input_dim, hid_dim, gln)

    + **Lastly**: Run the task, get the trained model and save it.
    .. code-block:: python

        pt.train(graph_list, batch_size=batch_size, lr=0.01, decay=0.0001, epochs=100)


- The following codes present a simple example on how to pre-train a GNN model via GraphCL:

.. code-block:: python

    from ProG.utils import mkdir, load_data4pretrain
    from ProG import PreTrain

    mkdir('./pre_trained_gnn/')

    args = get_args()

    print("load data...")
    input_dim, out_dim, graph_list = load4graph(args.dataset_name, pretrained=True)

    print("create PreTrain instance...")
    pt = GraphCL(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)

    print("pre-training...")
    pt.pretrain().

- Other pre-train methods can also be done in the similar way.

.. code-block:: python

    #SimGRACE
    pt = SimGRACE(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)

    #Edgepred_GPPT
    pt = Edgepred_GPPT(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)

    #DGI
    pt = DGI(dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=args.device)

    #......

    pt.pretrain()