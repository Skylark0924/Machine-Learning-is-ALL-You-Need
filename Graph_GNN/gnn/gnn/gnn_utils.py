import numpy as np
import pandas as pd
import scipy.io as sio
import os
from scipy.sparse import coo_matrix
from collections import namedtuple
import scipy.sparse as sp
SparseMatrix = namedtuple("SparseMatrix", "indices values dense_shape")

def GetInput(mat, lab, batch=1, grafi=None):
    """grafi is vector with same cardinaluty of nodes, denoting to which graph
        belongs each node
    """
    # numero di batch
    batch_number = grafi.max() // batch   # if only one graph => grafi.max() is 0 => batch_number == 0
    # dataframe containing adjacency matrix
    dmat = pd.DataFrame(mat, columns=["id_1", "id_2"])
    # dataframe containing labels each node
    dlab = pd.DataFrame(lab, columns=["lab" + str(i) for i in range(0, lab.shape[1])])
    # darch=pd.DataFrame(arc, columns=["arch"+str(i) for i in range(0,arc.shape[1])])
    # dataframe denoting graph belonging each node
    dgr = pd.DataFrame(grafi, columns=["graph"])

    # creating input : id_p, id_c, label_p, label_c, graph_belong
    dresult = dmat
    dresult = pd.merge(dresult, dlab, left_on="id_1", right_index=True, how='left')
    dresult = pd.merge(dresult, dlab, left_on="id_2", right_index=True, how='left')
    # dresult=pd.concat([dresult, darch], axis=1)
    dresult = pd.merge(dresult, dgr, left_on="id_1", right_index=True, how='left')

    data_batch = []
    arcnode_batch = []
    nodegraph_batch = []
    node_in = []
    # creating batch data => for each batch, redefining the id so that they start from 0 index
    for i in range(0, batch_number + 1):

        # getting minimum index of the current batch
        grafo_indexMin = (i * batch)
        grafo_indexMax = (i * batch) + batch

        adj = dresult.loc[(dresult["graph"] >= grafo_indexMin) & (dresult["graph"] < grafo_indexMax)]
        min_id = adj[["id_1", "id_2"]].min(axis=0).min()

        #start from 0 index for the new batch
        adj["id_1"] = adj["id_1"] - min_id
        adj["id_2"] = adj["id_2"] - min_id

        min_gr = adj["graph"].min()
        adj["graph"] = adj["graph"] - min_gr

        # append values to batches : id_2, lab0_1, lab1_1, lab0_2, lab1_2 (excluded first and last - id_p and graph_id)
        data_batch.append(adj.values[:, :-1])

        # arcMat creation

        # max_id of nodes in the current batch
        max_id = int(adj[["id_1", "id_2"]].max(axis=0).max())

        max_gr = int(adj["graph"].max())

        # getting ids of nodes (p and c)
        mt = adj[["id_1", "id_2"]].values
        # arcnode matrix : first shape same as arcs, second same as nodes in the batch
        arcnode = np.zeros((mt.shape[0], max_id + 1))

        # arcnode: state of parent node = sum (h(state of all the neighbors ,..) (of the parent node)
        # => sum contributes of all the arcs involving the parent
        # in j-th arc (row) => put one in the position corresponding to the parent node's column
        # => found in the adjacnecy matrix in j-th row, 1 st position

        # for j in range(0, mt.shape[0]):
        #     arcnode[j][mt[j][0]] = 1

        arcnode = SparseMatrix(indices=np.stack((mt[:, 0], np.arange(len(mt))), axis=1), values=np.ones([len(mt)]),
                               dense_shape=[max_id + 1, len(mt)])

        arcnode_batch.append(arcnode)

        # nodegraph
        # nodegraph = np.zeros((max_id + 1, max_gr + 1))

        # for t in range(0, max_id + 1):
        #     val = adj[["graph"]].loc[(adj["id_1"] == t) | (adj["id_2"] == t)].values[0]
        #     nodegraph[t][val] = 1

        nodegraph = SparseMatrix(indices=np.stack((dgr["graph"].values, np.arange(max_id+1)), axis=1), values=np.ones(max_id+1),
                               dense_shape=[max_gr+1, max_id + 1])


        nodegraph_batch.append(nodegraph)
        # node number in each graph
        grbtc = dgr.loc[(dgr["graph"] >= grafo_indexMin) & (dgr["graph"] < grafo_indexMax)]
        #counting number nodes in current batch
        node_in.append(grbtc.groupby(["graph"]).size().values)

    return data_batch, arcnode_batch, nodegraph_batch, node_in


def set_load_subgraph(data_path, set_type):
    # load adjacency list
    types = ["train", "valid", "test"]
    try:
        if set_type not in types:
            raise NameError('Wrong set name!')

        # load adjacency list
        mat = sio.loadmat(os.path.join(data_path, 'conmat{}.mat'.format(set_type)))
        # load adiacenyc matrixc in sparse format
        adj = coo_matrix(mat["conmat_{}set".format(set_type)].T)
        adj = np.array([adj.row, adj.col]).T

        # load node label
        mat = sio.loadmat(os.path.join(data_path, "nodelab{}.mat".format(set_type)))
        lab = np.asarray(mat["nodelab_{}set".format(set_type)]).T

        # load target and convert to one-hot encoding
        mat = sio.loadmat(os.path.join(data_path, "tar{}.mat".format(set_type)))
        target = np.asarray(mat["target_{}set".format(set_type)]).T
        # one-hot encoding of targets
        labels = pd.get_dummies(pd.Series(target.reshape(-1)))
        labels = labels.values
        # compute inputs and arcnode
        inp, arcnode, nodegraph, nodein = GetInput(adj, lab, 1, np.zeros(len(labels), dtype=int)) # last argument: graph to which each node belongs
        return inp, arcnode, nodegraph, nodein, labels, lab

    except Exception as e:
        print("Caught exception: ", e)
        exit(1)

def set_load_clique(data_path, set_type):
    import load as ld
    # load adjacency list
    types = ["train", "validation", "test"]
    train = ld.loadmat(os.path.join(data_path, "cliquedataset.mat"))
    train = train["dataSet"]
    try:
        if set_type not in types:
            raise NameError('Wrong set name!')

        # load adjacency list
        # take adjacency list
        adj = coo_matrix(train['{}Set'.format(set_type)]['connMatrix'].T)
        adj = np.array([adj.row, adj.col]).T

        # take node labels
        lab = np.asarray(train['{}Set'.format(set_type)]['nodeLabels']).T

        # take targets and convert to one-hot encoding
        target = np.asarray(train['{}Set'.format(set_type)]['targets']).T
        labels = pd.get_dummies(pd.Series(target))
        labels = labels.values

        # compute inputs and arcnode
        get_lab = lab.reshape(lab.shape[0], 1) if set_type == "train" else lab.reshape(len(labels), 1)
        inp, arcnode, nodegraph, nodein = GetInput(adj, get_lab, 1,
                                                           np.zeros(len(labels), dtype=int))
        return inp, arcnode, nodegraph, nodein, labels

    except Exception as e:
        print("Caught exception: ", e)
        exit(1)


def set_load_mutag(set_type, train):
    # load adjacency list
    types = ["train", "validation", "test"]
    try:
        if set_type not in types:
            raise NameError('Wrong set name!')

            ############ training set #############

            # take adjacency list
        adj = coo_matrix(train['{}Set'.format(set_type)]['connMatrix'])
        adj = np.array([adj.row, adj.col]).T

        # take node labels
        lab = np.asarray(train['{}Set'.format(set_type)]['nodeLabels']).T
        mask = coo_matrix(train['{}Set'.format(set_type)]["maskMatrix"])

        # take target, generate output for each graph, and convert to one-hot encoding
        target = np.asarray(train['{}Set'.format(set_type)]['targets']).T
        v = mask.col
        target = np.asarray([target[x] for x in v])
        # target = target[target != 0] # equivalent code
        labels = pd.get_dummies(pd.Series(target))
        labels = labels.values

        # build graph indices
        gr = np.array(mask.col)
        indicator = []
        for j in range(0, len(gr) - 1):
            for i in range(gr[j], gr[j + 1]):
                indicator.append(j)
        for i in range(gr[-1], adj.max() + 1):
            indicator.append(len(gr) - 1)
        indicator = np.asarray(indicator)

        # take input, arcnode matrix, nodegraph matrix
        inp, arcnode, nodegraph, nodein = GetInput(adj, lab, indicator.max() + 1, indicator)

        return inp, arcnode, nodegraph, nodein, labels

    except Exception as e:
        print("Caught exception: ", e)
        exit(1)


def set_load_general(data_path, set_type, set_name="sub_30_15"):
    import load as ld
    # load adjacency list
    types = ["train", "validation", "test"]
    train = ld.loadmat(os.path.join(data_path, "{}.mat".format(set_name)))
    train = train["dataSet"]
    try:
        if set_type not in types:
            raise NameError('Wrong set name!')

        # load adjacency list
        # take adjacency list
        adj = coo_matrix(train['{}Set'.format(set_type)]['connMatrix'].T)
        adj = np.array([adj.row, adj.col]).T

        # take node labels
        lab = np.asarray(train['{}Set'.format(set_type)]['nodeLabels']).T

        # if clique (labels with only one dimension
        if len(lab.shape) < 2:
            lab = lab.reshape(lab.shape[0], 1)

        # take targets and convert to one-hot encoding
        target = np.asarray(train['{}Set'.format(set_type)]['targets']).T
        labels = pd.get_dummies(pd.Series(target))
        labels = labels.values

        # compute inputs and arcnode

        inp, arcnode, nodegraph, nodein = GetInput(adj, lab, 1,
                                                           np.zeros(len(labels), dtype=int))
        return inp, arcnode, nodegraph, nodein, labels, lab

    except Exception as e:
        print("Caught exception: ", e)
        exit(1)


def load_karate(path="data/karate-club/"):
    """Load karate club dataset"""
    print('Loading karate club dataset...')
    import random

    edges = np.loadtxt("{}edges.txt".format(path), dtype=np.int32) - 1  # 0-based indexing
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]  # reorder list of edges also by second column
    features = sp.eye(np.max(edges+1), dtype=np.float32).tocsr()

    idx_labels = np.loadtxt("{}mod-based-clusters.txt".format(path), dtype=np.int32)
    idx_labels = idx_labels[idx_labels[:, 0].argsort()]

    labels = np.eye(max(idx_labels[:, 1])+1, dtype=np.int32)[idx_labels[:, 1]]  # one-hot encoding of labels

    E = np.concatenate((edges, np.zeros((len(edges), 1), dtype=np.int32)), axis=1)
    N = np.concatenate((features.toarray(), np.zeros((features.shape[0], 1), dtype=np.int32)), axis=1)

    mask_train = np.zeros(shape=(34,), dtype=np.float32)
    idx_classes = np.argmax(labels, axis=1)

    id_0, id_4, id_5, id_12 = random.choices(np.argwhere(idx_classes == 0), k=4)
    id_1, id_6, id_7, id_13 = random.choices(np.argwhere(idx_classes == 1), k=4)
    id_2, id_8, id_9, id_14 = random.choices(np.argwhere(idx_classes == 2), k=4)
    id_3, id_10, id_11, id_15 = random.choices(np.argwhere(idx_classes == 3), k=4)

    mask_train[id_0] = 1.  # class 1
    mask_train[id_1] = 1.  # class 2
    mask_train[id_2] = 1.  # class 0
    mask_train[id_3] = 1.  # class 3
    mask_test = 1. - mask_train

    return E, N, labels, mask_train, mask_test


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