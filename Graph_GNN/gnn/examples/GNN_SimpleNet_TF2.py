import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import gnn.gnn_utils as gnn_utils
# import gnn.GNN as GNN
# from examples import Net_Simple

# import networkx as nx
import scipy as sp

import matplotlib.pyplot as plt

##### GPU & stuff config
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

############# data creation ################

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

E = np.concatenate((E, np.asarray(e2)), axis=0)

N_tot = np.eye(edges + edges_2,  dtype=np.float32)
N_tot = np.concatenate((N_tot, np.zeros((edges + edges_2,1), dtype=np.float32)),  axis=1 )
print(N_tot)

# Create Input to GNN

import numpy as np
import pandas as pd
import scipy.io as sio
import os
from scipy.sparse import coo_matrix
from collections import namedtuple
import scipy.sparse as sp
SparseMatrix = namedtuple("SparseMatrix", "indices values dense_shape")
def from_EN_to_GNN(E, N):
    """
    :param E: # E matrix - matrix of edges : [[id_p, id_c, graph_id],...]
    :param N: # N matrix - [node_features, graph_id (to which the node belongs)]
    :return: # L matrix - list of graph targets [tar_g_1, tar_g_2, ...]
    """
    N_full = N
    N = N[:, :-1]  # avoid graph_id
    e = E[:, :2]  # take only first tow columns => id_p, id_c
    feat_temp = np.take(N, e, axis=0)  # take id_p and id_c  => (n_archs, 2, label_dim)
    feat = np.reshape(feat_temp, [len(E), -1])  # (n_archs, 2*label_dim) => [[label_p, label_c], ...]
    # creating input for gnn => [id_p, id_c, label_p, label_c]
    inp = np.concatenate((E[:, :2], feat), axis=1)
    # creating arcnode matrix, but transposed
    """
    1 1 0 0 0 0 0 
    0 0 1 1 0 0 0
    0 0 0 0 1 1 1    
    """  # for the indices where to insert the ones, stack the id_p and the column id (single 1 for column)
    arcnode = SparseMatrix(indices=np.stack((E[:, 0], np.arange(len(E))), axis=1),
                           values=np.ones([len(E)]).astype(np.float32),
                           dense_shape=[len(N), len(E)])

    # get the number of graphs => from the graph_id
    num_graphs = int(max(N_full[:, -1]) + 1)
    # get all graph_ids
    g_ids = N_full[:, -1]
    g_ids = g_ids.astype(np.int32)

    # creating graphnode matrix => create identity matrix get row corresponding to id of the graph
    # graphnode = np.take(np.eye(num_graphs), g_ids, axis=0).T
    # substitued with same code as before
    graphnode = SparseMatrix(indices=np.stack((g_ids, np.arange(len(g_ids))), axis=1),
                             values=np.ones([len(g_ids)]).astype(np.float32),
                             dense_shape=[num_graphs, len(N)])

    # print(graphnode.shape)

    return inp, arcnode, graphnode

inp, arcnode, graphnode = from_EN_to_GNN(E, N_tot)
labels = np.random.randint(2, size=(N_tot.shape[0]))


labels = np.eye(max(labels)+1, dtype=np.int32)[labels]  # one-hot encoding of labels


################################################################################################
################################################################################################
################################################################################################
################################################################################################

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
threshold = 0.01
learning_rate = 0.01
state_dim = 5
input_dim = inp.shape[1]
output_dim = labels.shape[1]
max_it = 50
num_epoch = 10000
# optimizer = tf.train.AdamOptimizer

# initialize state and output network
# net = n.Net(input_dim, state_dim, output_dim)

# initialize GNN
param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
print(param)



import tensorflow as tf
import numpy as np

class Net(tf.keras.Model):
    def __init__(self, ArcNode, input_dim, state_dim, output_dim):
        super(Net,self).__init__(name='')
        # initialize weight and parameter
        
        self.ArcNode = tf.sparse.SparseTensor(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        

        self.state_threshold = 0.01
        
        self.max_iter = 50

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.state_input = self.input_dim - 1 + state_dim  # removing the id_ dimension

        #### TO BE SET ON A SPECIFIC PROBLEM
        self.state_l1 = 15
        self.state_l2 = self.state_dim

        self.output_l1 = 10
        self.output_l2 = self.output_dim

        self.k = tf.Variable(0,name='k')
        
        self.state = tf.Variable(state_init, name="state",dtype=tf.float32)

        self.state_old = tf.Variable(state_init, name="old_state",dtype=tf.float32)
        
        self.layer_1_state = tf.keras.layers.Dense(self.state_l1, activation='tanh')
        self.layer_2_state = tf.keras.layers.Dense(self.state_l2, activation='tanh')
        self.layer_1_out = tf.keras.layers.Dense(self.output_l1, activation='tanh')
        self.layer_2_out = tf.keras.layers.Dense(self.output_l2, activation='softmax')
        

    def convergence(self, a, state, old_state, k):

        # assign current state to old state
        old_state = state

        # grub states of neighboring node
        gat = tf.gather(old_state, tf.cast(a[:, 1], tf.int32))

        # slice to consider only label of the node and that of it's neighbor
        # sl = tf.slice(a, [0, 1], [tf.shape(a)[0], tf.shape(a)[1] - 1])
        # equivalent code
        sl =a[:, 2:]

        # concat with retrieved state
        inp = tf.concat([sl, gat], axis=1)

        # evaluate next state and multiply by the arch-node conversion matrix to obtain per-node states
        layer_1_state = self.layer_1_state(inp)
        layer_2_state = self.layer_2_state(layer_1_state)

        state = tf.sparse.sparse_dense_matmul(self.ArcNode, layer_2_state)

        # update the iteration counter
        k = k + 1
        return a, state, old_state, k

    def condition(self, a, state, old_state, k):
        # evaluate condition on the convergence of the state

        # evaluate distance by state(t) and state(t-1)
        outDistance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, old_state)), 1) + 0.00000000001)
        # vector showing item converged or not (given a certain threshold)
        checkDistanceVec = tf.greater(outDistance, self.state_threshold)

        c1 = tf.reduce_any(checkDistanceVec)
        c2 = tf.less(k, self.max_iter)

        return tf.logical_and(c1, c2)
        

    def call(self, comp_inp):
        
        k = tf.constant(0)
        res, st, old_st, num = tf.while_loop(self.condition, self.convergence,
                                                 [comp_inp, self.state, self.state_old, k])
        
        
        layer_1_out = self.layer_1_out(st)
        layer_2_out = self.layer_2_out(layer_1_out)
#         print(layer_2_out,comp_inp.shape)

        
        return layer_2_out,num
    

EPSILON = 0.00000001

@tf.function()
def custom_loss(target,output):
    target = tf.cast(target,tf.float32)
    output = tf.maximum(output, EPSILON, name="Avoiding_explosions")  # to avoid explosions
    xent = -tf.reduce_sum(target * tf.math.log(output), 1)
    lo = tf.reduce_mean(xent)
    return lo

@tf.function()
def metric(output, target):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
    metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return metric
	
	
def create_model():

    comp_inp = tf.keras.Input(shape=(input_dim), name="input")
    

    output,_ = Net(arcnode,input_dim, state_dim, output_dim)(comp_inp)
    
    model = tf.keras.Model(comp_inp, output)

    return model
	
model = create_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
for _ in range(10):

    with tf.GradientTape() as tape:
        
        out = model(inp,training=True)

        loss_value = custom_loss(labels,out)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        print(loss_value)
