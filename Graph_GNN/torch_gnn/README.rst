A PyTorch implementation of the Graph Neural Network Model
==========================================================

This repo contains a PyTorch implementation of the Graph Neural Network model.

The `main_simple.py <https://github.com/mtiezzi/torch_gnn/blob/master/main_simple.py>`_ example shows how to use the `EN_input <https://mtiezzi.github.io/gnn_site/PyTorch.html#en-input>`_ format. 


Have a look at the Subgraph Matching/Clique detection example, contained in the file `main_subgraph.py <https://github.com/mtiezzi/torch_gnn/blob/master/main_subgraph.py>`_.

An example of handling the Karate Club dataset can be found in the example `main_enkarate.py <https://github.com/mtiezzi/torch_gnn/blob/master/main_enkarate.py>`_.


- **Website (including documentation):** https://mtiezzi.github.io/gnn_site/
- **Author:** `Matteo Tiezzi <http://mtiezzi.github.io/>`_  
Install
-------

Requirements
^^^^^^^^^^^^
The GNN framework requires the packages **PyTorch**, **numpy**, **scipy**.


To install the requirements you can use the following command
::


      pip install -U -r requirements.txt



For additional details, please see `Install <https://mtiezzi.github.io/gnn_site/install.html>`_.

Simple usage example
--------------------

::

        import torch
        import utils
        import dataloader
        from gnn_wrapper import GNNWrapper, SemiSupGNNWrapper
        
        # define GNN configuration 
        cfg = GNNWrapper.Config()
        cfg.use_cuda = use_cuda
        cfg.device = device       

        cfg.activation = nn.Tanh()
        cfg.state_transition_hidden_dims = [5,]
        cfg.output_function_hidden_dims = [5]
        cfg.state_dim = 2
        cfg.max_iterations = 50
        cfg.convergence_threshold = 0.01
        cfg.graph_based = False
        cfg.task_type = "semisupervised"
        cfg.lrw = 0.001

        model = SemiSupGNNWrapper(cfg)
        # Provide your own functions to generate input data
        E, N, targets, mask_train, mask_test = dataloader.old_load_karate()
        dset = dataloader.from_EN_to_GNN(E, N, targets, aggregation_type="sum", sparse_matrix=True)  # generate the dataset

        # Create the state transition function, output function, loss function and  metrics 
        net = n.Net(input_dim, state_dim, output_dim)

        
        
        #Training
                
        for epoch in range(args.epochs):
            model.train_step(epoch)



Citing
------

To cite the GNN implementation please use the following publication::

    Matteo Tiezzi, Giuseppe Marra, Stefano Melacci, Marco Maggini and Marco Gori (2020). "A Lagrangian Approach to Information Propagation in Graph Neural Networks; ECAI2020

Bibtex::

    @article{tiezzi2020lagrangian,
      title={A Lagrangian Approach to Information Propagation in Graph Neural Networks},
      author={Tiezzi, Matteo and Marra, Giuseppe and Melacci, Stefano and Maggini, Marco and Gori, Marco},
      journal={arXiv preprint arXiv:2002.07684},
      year={2020}
    }


License
-------

Released under the 3-Clause BSD license (see `LICENSE.txt`)::

   Copyright (C) 2004-2020 Matteo Tiezzi
   Matteo Tiezzi <mtiezzi@diism.unisi.it>
