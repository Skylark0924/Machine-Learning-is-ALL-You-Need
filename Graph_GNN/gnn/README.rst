Graph Neural Network Model
==========================

This repo contains a Tensorflow implementation of the Graph Neural Network model.


- **Website (including documentation):** https://mtiezzi.github.io/gnn_site/
- **Authors:** `Matteo Tiezzi <http://sailab.diism.unisi.it/people/matteo-tiezzi/>`_, `Alberto Rossi <http://sailab.diism.unisi.it/people/alberto-rossi/>`_

Install
-------

Requirements
^^^^^^^^^^^^
The GNN framework requires the packages **tensorflow**, **numpy**, **scipy**.


To install the requirements you can use the following command
::


      pip install -U -r requirements.txt


Install the latest version of GNN::

      pip install gnn


For additional details, please see `Install <https://mtiezzi.github.io/gnn_site/install.html>`_.

Simple usage example
--------------------

::

        import gnn.GNN as GNN
        import gnn.gnn_utils
        import Net as n
        
        # Provide your own functions to generate input data
        inp, arcnode, nodegraph, labels = set_load()

        # Create the state transition function, output function, loss function and  metrics 
        net = n.Net(input_dim, state_dim, output_dim)

        # Create the graph neural network model
        g = GNN.GNN(net, input_dim, output_dim, state_dim)
        
        #Training
                
        for j in range(0, num_epoch):
            g.Train(inp, arcnode, labels, count, nodegraph)
            
            # Validate            
            print(g.Validate(inp_val, arcnode_val, labels_val, count, nodegraph_val))


Citing
------

To cite the GNN implementation please use the following publication::

    Rossi, A., Tiezzi, M., Dimitri, G.M., Bianchini, M., Maggini, M., & Scarselli, F. (2018).
    "Inductive–Transductive Learning with Graph Neural Networks", 
    In Artificial Neural Networks in Pattern Recognition (pp.201-212). 
    Berlin : Springer-Verlag.

Bibtex::

    @inproceedings{rossi2018inductive,
      title={Inductive--Transductive Learning with Graph Neural Networks},
      author={Rossi, Alberto and Tiezzi, Matteo and Dimitri, Giovanna Maria and Bianchini, Monica and Maggini, Marco and Scarselli, Franco},
      booktitle={IAPR Workshop on Artificial Neural Networks in Pattern Recognition},
      pages={201--212},
      year={2018},
      organization={Springer}
    }


To cite GNN please use the following publication::

    F. Scarselli, M. Gori,  A. C. Tsoi, M. Hagenbuchner, G. Monfardini, 
    "The Graph Neural Network Model", IEEE Transactions on Neural Networks,  
    vol. 20(1); p. 61-80, 2009.

Bibtex::

    @article{Scarselli2009TheGN,
      title={The Graph Neural Network Model},
      author={Franco Scarselli and Marco Gori and Ah Chung Tsoi and Markus Hagenbuchner and Gabriele Monfardini},
      journal={IEEE Transactions on Neural Networks},
      year={2009},
      volume={20},
      pages={61-80}
    }



Contributions
-------------

In the example folder, file  `GNN_SimpleNet_TF2.py <https://github.com/sailab-code/gnn/blob/master/examples/GNN_SimpleNet_TF2.py>`_ you can find a tentative all-in-one implementation in Tensorflow 2, a contribution by `Rohan Kotwani <https://github.com/freedomtowin>`_ .
We thank him and all the interested users!


You can find a TF 2.x implementation by  N.Pancino and P.Bongini (PhD Students @ SAILab) at this repo  `repo  <https://github.com/sailab-code/GNN_tf_2.x>`_


License
-------

Released under the 3-Clause BSD license (see `LICENSE.txt`)::

   Copyright (C) 2004-2019 Matteo Tiezzi
   Matteo Tiezzi <mtiezzi@diism.unisi.it>
   Alberto Rossi <alrossi@unifi.it>
