import torch
import torch.nn as nn
import numpy as np
import argparse
import utils
import sys
sys.path.append('/home/skylark/Github/Machine-Learning-Basic-Codes/Graph_GNN/torch_gnn')
import dataloader
from gnn_wrapper import GNNWrapper, SemiSupGNNWrapper
import matplotlib.pyplot as plt
import networkx as nx


# GRAPH #1

# List of edges in the first graph - last column is the id of the graph to which the arc belongs
e = [[0, 1, 0], [0,2, 0], [0, 4, 0], [1, 2, 0], [1, 3, 0], [2, 3, 0], [2, 4, 0]]
# undirected graph, adding other direction
e.extend([[i, j, num] for j, i, num in e])
#reorder
e = sorted(e)
E = np.asarray(e)

#number of nodes
edges = 5
# creating node features - simply one-hot values
N = np.eye(edges, dtype=np.float32)

# adding column thta represent the id of the graph to which the node belongs
N = np.concatenate((N, np.zeros((edges,1), dtype=np.float32)),  axis=1 )


# visualization graph
def plot_graph(E, N):
    g = nx.Graph()
    g.add_nodes_from(range(N.shape[0]))
    g.add_edges_from(E[:, :2])
    nx.draw_spring(g, cmap=plt.get_cmap('Set1'), with_labels=True)
    plt.show()


plot_graph(E,N)



# GRAPH #2

# List of edges in the second graph - last column graph-id
e1 = [[0, 2, 1], [0,3,1], [1, 2,1], [1,3,1], [2,3,1]]
# undirected graph, adding other direction
e1.extend([[i, j, num] for j, i, num in e1])
# reindexing node ids based on the dimension of previous graph (using unique ids)
e2 = [[a + N.shape[0], b + N.shape[0], num] for a, b, num in e1]
#reorder
e2 = sorted(e2)
edges_2 = 4


# Plot second graph

E1 = np.asarray(e1)
N1 = np.eye(edges_2,  dtype=np.float32)
N1 = np.concatenate((N1, np.zeros((edges_2,1), dtype=np.float32)),  axis=1 )

plot_graph(E1,N1)

E = np.concatenate((E, np.asarray(e2)), axis=0)
N_tot = np.eye(edges + edges_2,  dtype=np.float32)
N_tot = np.concatenate((N_tot, np.zeros((edges + edges_2,1), dtype=np.float32)),  axis=1 )

# Create Input to GNN

labels = np.random.randint(2, size=(N_tot.shape[0]))
#labels = np.eye(max(labels)+1, dtype=np.int32)[labels]  # one-hot encoding of labels


cfg = GNNWrapper.Config()
cfg.use_cuda = True
cfg.device = utils.prepare_device(n_gpu_use=1, gpu_id=0)
cfg.tensorboard = False
cfg.epochs = 500

cfg.activation = nn.Tanh()
cfg.state_transition_hidden_dims = [5,]
cfg.output_function_hidden_dims = [5]
cfg.state_dim = 5
cfg.max_iterations = 50
cfg.convergence_threshold = 0.01
cfg.graph_based = False
cfg.log_interval = 10
cfg.task_type = "multiclass"
cfg.lrw = 0.001

# model creation
model = GNNWrapper(cfg)
# dataset creation
dset = dataloader.from_EN_to_GNN(E, N_tot, targets=labels, aggregation_type="sum", sparse_matrix=True)  # generate the dataset

model(dset)  # dataset initalization into the GNN

# training code
for epoch in range(1, cfg.epochs + 1):
    model.train_step(epoch)

    if epoch % 10 == 0:
        model.test_step(epoch)
