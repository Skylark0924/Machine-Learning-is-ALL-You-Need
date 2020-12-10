import tensorflow as tf
import numpy as np
import utils
import GNNs as GNN
import Net_Karate as n
from scipy.sparse import coo_matrix

##### GPU & stuff config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

############# training set ################


E, N, labels,  mask_train, mask_test = utils.load_karate()
inp, arcnode, graphnode = utils.from_EN_to_GNN(E, N)

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
threshold = 0.001
learning_rate = 0.001
state_dim = 2
tf.reset_default_graph()
input_dim = inp.shape[1]
output_dim = labels.shape[1]
max_it = 50
num_epoch = 10000
optimizer = tf.train.AdamOptimizer

# initialize state and output network
net = n.Net(input_dim, state_dim, output_dim)

# initialize GNN
param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
print(param)

tensorboard = False

g = GNN.GNN(net, input_dim, output_dim, state_dim,  max_it, optimizer, learning_rate, threshold, graph_based=False, param=param, config=config,
            tensorboard=tensorboard, mask_flag=True)

# train the model
count = 0

######

for j in range(0, num_epoch):
    _, it = g.Train(inputs=inp, ArcNode=arcnode, target=labels, step=count, mask=mask_train)

    if count % 10 == 0:
        print("Epoch ", count)
        print("Training: ", g.Validate(inp, arcnode, labels, count, mask=mask_train))
        print("Test: ", g.Validate(inp, arcnode, labels, count, mask=mask_test))

        # end = time.time()
        # print("Epoch {} at time {}".format(j, end-start))
        # start = time.time()

    count = count + 1

# evaluate on the test set
#print("\nEvaluate: \n")
#print(g.Evaluate(inp_test[0], arcnode_test[0], labels_test, nodegraph_test[0])[0])
