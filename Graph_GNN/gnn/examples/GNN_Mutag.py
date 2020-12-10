import numpy as np
import gnn.gnn_utils as gnn_utils
import gnn.GNN as GNN
import Net_Mutag as n
import tensorflow as tf
import load as ld
from scipy.sparse import coo_matrix
import os



os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#############   DATA LOADING    ##################################################
# function to get a fold
def getFold(fold):
    # load dataset
    train = ld.loadmat("./Data/Mutag/multi" + str(fold))
    train = train['multi' + str(fold)]

    ############ training set #############

    ret_train = gnn_utils.set_load_mutag("train", train)

    ###########validation#####################

    ret_val = gnn_utils.set_load_mutag("validation", train)

    ########### test #####################

    ret_test = gnn_utils.set_load_mutag("test", train)

    return ret_train, ret_val, ret_test


# create the 10-fold in order to train on 10-fold cross validation
tr, val, ts = [], [], []
for fold in range(1, 11):
    a, b, c = getFold(fold)
    tr.append(a)
    val.append(b)
    ts.append(c)

# set parameter
threshold = 0.001
learning_rate = 0.0001
state_dim = 5
max_it = 50
num_epoch = 1000
optimizer = tf.train.AdamOptimizer

output_dim = 2

testacc = []

for fold in range(0, 10):

    tf.reset_default_graph()
    param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
    completeName = param + 'log.txt'
    param = param + "_fold" + str(fold)
    print(param)


    # retrieve input, arcnode, nodegraph and target for training set
    inp = tr[fold][0]
    input_dim = len(inp[0][0])

    arcnode = tr[fold][1]
    labels = tr[fold][4]
    nodegraph = tr[fold][2]

    # retrieve input, arcnode, nodegraph and target for validation set
    inp_val = val[fold][0]
    arcnode_val = val[fold][1]
    labels_val = val[fold][4]
    nodegraph_val = val[fold][2]

    # initialize network
    net = n.Net(input_dim, state_dim, output_dim)


    # instantiate GNN
    g = GNN.GNN(net, input_dim, output_dim, state_dim,  max_it,  optimizer, learning_rate, threshold, graph_based=True,
                param=param, config=config)

    # train GNN, and validate every 2 epochs, (early stopping)
    count = 0
    valid_best = None
    patience = 0
    for j in range(0, num_epoch):
        g.Train(inp[0], arcnode[0], labels, count, nodegraph[0])
        print("Epoch ", j)
        if count % 2 == 0:

            loss = g.Validate(inp_val[0], arcnode_val[0], labels_val, count, nodegraph_val[0])
            if count == 0:
                valid_best = loss

            if loss < valid_best:
                valid_best = loss
                #save_path = g.saver.save(g.session, g.save_path)
                patience = 0
            else:
                patience += 1

            if patience > 5:
                print("Early stopping...")
                break
        count = count + 1

    # retrieve input, arcnode, nodegraph and target for test set
    inp_test = ts[fold][0]
    arcnode_test = ts[fold][1]
    labels_test = ts[fold][4]
    nodegraph_test = ts[fold][2]
    print('Accuracy on test set fold ', fold, ' :')

    # evaluate on the test set fold
    evel = g.Evaluate(inp_test[0], arcnode_test[0], labels_test, nodegraph_test[0])
    testacc.append(evel)
    print(evel)
    with open(os.path.join('tmp/', completeName), "a") as file:
        file.write('Accuracy on test set fold ' + str(fold) + ' :')
        file.write(str(evel) + '\n')
        file.write('\n')
        file.close()

# mean accuracy on the 10-fold
mean_acc = np.mean(np.asarray(testacc))
print('Mean accuracy from all folds:', mean_acc)
