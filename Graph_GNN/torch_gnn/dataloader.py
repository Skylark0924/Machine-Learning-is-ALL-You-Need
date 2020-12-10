import numpy as np
from abc import abstractmethod
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
import networkx as nx
import typing
import scipy
import scipy.io as spio
import numpy as np
import os


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''

    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


# train = loadmat('multi1')

# thanks Pedro H. Avelar
def nx_to_format(G, aggregation_type, sparse_matrix=True):
    e = len(G.edges)
    n = len(G.nodes)

    # edges = torch.LongTensor(list(G.edges))
    edg = sorted(list(G.edges))
    edges = torch.LongTensor(edg)

    adj_matrix = np.asarray(nx.to_numpy_matrix(G))

    if aggregation_type == "sum":
        pass
    elif aggregation_type == "degreenorm":
        row_sum = np.sum(adj_matrix, axis=0, keepdims=True)
        adj_matrix = adj_matrix / row_sum
    elif aggregation_type == "symdegreenorm":
        raise NotImplementedError("Symmetric degree normalization not yet implemented")
    else:
        raise ValueError("Invalid neighbour aggregation type")

    if sparse_matrix:
        agg_matrix_i = torch.LongTensor([[s for s, t in G.edges], list(range(e))])
        agg_matrix_v = torch.FloatTensor([adj_matrix[s, t] for s, t in G.edges])
        agg_matrix = torch.sparse.FloatTensor(agg_matrix_i, agg_matrix_v, torch.Size([n, e]))
    else:
        agg_matrix = torch.zeros(*[n, e])
        for i, (s, t) in enumerate(edg):
            agg_matrix[s, i] = adj_matrix[s, t]

    return edges, agg_matrix




class Dataset:
    def __init__(
            self,
            name,
            num_nodes,
            num_edges,
            label_dim,
            is_multiclass,
            num_classes,
            edges,
            agg_matrix,
            node_labels,
            targets,
            idx_train=None,
            idx_valid=None,
            idx_test=None,
            graph_node=None
    ):
        self.name = name
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_label_dim = label_dim
        self.num_classes = num_classes
        self.is_multiclass = is_multiclass
        self.edges = edges
        self.agg_matrix = agg_matrix
        self.node_labels = node_labels
        self.targets = targets
        self.idx_train = idx_train
        self.idx_valid = idx_valid
        self.idx_test = idx_test

    def cuda(self):
        self.edges, self.agg_matrix, self.node_labels, self.targets, self.idx_train, self.idx_test = map(
            lambda x: x.cuda() if x is not None else None,
            [self.edges, self.agg_matrix, self.node_labels, self.targets, self.idx_train, self.idx_test]
        )
        return self

    def cpu(self):
        self.edges, self.agg_matrix, self.node_labels, self.targets, self.idx_train, self.idx_test = map(
            lambda x: x.cuda(),
            [self.edges, self.agg_matrix, self.node_labels, self.targets, self.idx_train, self.idx_test]
        )
        return self

    def to(self, device):
        if "cuda" in device.type:
            torch.cuda.set_device(device)
            return self.cuda()
        else:
            return self.cpu()


def get_twochains(num_nodes_per_graph=50, pct_labels=.1, pct_valid=.5, aggregation_type="sum", sparse_matrix=True):
    G1 = nx.generators.classic.path_graph(num_nodes_per_graph)
    G2 = nx.generators.classic.path_graph(num_nodes_per_graph)

    G = nx.disjoint_union(G1, G2)
    G = G.to_directed()

    e = len(G.edges)
    n = len(G.nodes)

    edges, agg_matrix = nx_to_format(G, aggregation_type, sparse_matrix)

    is_multilabel = False
    n_classes = 2
    d_l = 1
    node_labels = torch.zeros(*[n, d_l])
    # node_labels = torch.eye(n)
    targets = torch.tensor(np.array(([0] * (n // 2)) + ([1] * (n // 2)), dtype=np.int64), dtype=torch.long)

    idx = np.random.permutation(np.arange(n))
    idx_trainval = idx[:int(n * pct_labels)]
    idx_train = torch.LongTensor(idx_trainval[:-int(len(idx_trainval) * pct_valid)])
    idx_valid = torch.LongTensor(
        idx_trainval[-int(len(idx_trainval) * pct_valid):])  # TODO wht is he doing, why with BoolTensro is strange?
    idx_test = torch.LongTensor(idx[int(n * pct_labels):])

    return Dataset(
        "twochains",
        n,
        e,
        d_l,
        is_multilabel,
        n_classes,
        edges,
        agg_matrix,
        node_labels,
        targets,
        idx_train,
        idx_valid,
        idx_test,
    )


############## SSE ################

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def read_sse_ids(percentage=None, dataset=None):
    def _internal(file):
        ids = []
        with open(os.path.join(dataset, file), 'r') as f:
            for line in f:
                ids.append(int(line.strip()))
        return ids

    if percentage:
        train_ids = _internal(
            "train_idx-{}.txt".format(
                percentage))  # list, each element a row of the file => id of the graph belonging to train set
        test_ids = _internal("test_idx-{}.txt".format(percentage))

    return train_ids, test_ids

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_twochainsSSE(aggregation_type, percentage=0.9, dataset="data/n-chains-connect", node_has_feature=False,
                     train_file="train_idx-", test_file="test_idx-", sparse_matrix=True):
    import os
    print('Loading dataset: {}'.format(dataset))
    graph_info = "meta.txt"
    neigh = "adj_list.txt"
    labels_file = "label.txt"
    # loading targets

    targets = np.loadtxt(os.path.join(dataset, labels_file))
    targets = torch.tensor(np.argmax(targets, axis=1), dtype=torch.long)

    with open(os.path.join(dataset, graph_info), 'r') as f:
        info = f.readline().strip().split()  # (ex. MUTAG - 23 2) number of nodes in the graph, target of the graph
        if node_has_feature:
            n_nodes, l, n_feat = [int(w) for w in info]  # n == number of nodes, l label (target) of the graph
        else:
            n_nodes, l = [int(w) for w in info]  # n == number of nodes, l label (target) of the graph
    # load adj_list
    if node_has_feature:
        features = np.loadtxt(os.path.join(dataset, "features.txt"))
    else:
        features = np.zeros((n_nodes, 1), dtype=np.float32)  # zero feature else

    with open(os.path.join(dataset, neigh), 'r') as f:

        g = nx.Graph()  # netxgraph
        node_features = []
        # n_edges = 0  # edges in the graph
        for j in range(n_nodes):
            # for every row of the current graph  create the graph itself
            g.add_node(j)  # add node to networkx graph

            row = [int(w) for w in
                   f.readline().strip().split()]  # composition of each row : number of neighbors, id_neigh_1, id_neigh_2 ...
            n_edges = row[0]  # increment edge counter with number of neighbors => number of arcs
            for k in range(1, n_edges + 1):
                g.add_edge(j, row[k])  # add edge in graph to all nodes from current one

        g = g.to_directed()  # every arc  # in this example, state of
        # e = [list(pair) for pair in g.edges()]  # [[0, 1], [0, 5], [1, 2], ... list containing lists of edge pair

        edges, agg_matrix = nx_to_format(g, aggregation_type, sparse_matrix)
        e = len(g.edges)
        n = len(g.nodes)
        d_l = 1
        is_multilabel = False
        n_classes = 2
        node_labels = torch.tensor(features, dtype=torch.float)
        # targets = torch.tensor(np.clip(target, 0, 1), dtype=torch.long)  # convert -1 to 0

        # creation of N matrix - [node_features, graph_id (to which the node belongs)] #here there is a unique graph
        # create mask for training
        train_ids, test_ids = read_sse_ids(percentage=percentage, dataset=dataset)
        # train_mask = sample_mask(train_ids, n)
        test_ids_temp = range(0, 2000)
        test_ids = [i for i in test_ids_temp if i not in train_ids]
        idx_train = torch.LongTensor(train_ids)
        idx_test = torch.LongTensor(test_ids)
        idx_valid = torch.LongTensor(test_ids)

        return Dataset(
            "two_chainsSSE",
            n,
            e,
            d_l,
            is_multilabel,
            n_classes,
            edges,
            agg_matrix,
            node_labels,
            targets,
            idx_train,
            idx_valid,
            idx_test,
        )


def get_subgraph(set="sub_10_5_200", aggregation_type="sum", sparse_matrix=False):
    from scipy.sparse import coo_matrix
    import scipy.sparse as sp
    import pandas as pd

    types = ["train", "validation", "test"]
    set_name = set
    train = loadmat("./data/subcli/{}.mat".format(set_name))
    train = train["dataSet"]
    dset = {}
    for set_type in types:
        adj = coo_matrix(train['{}Set'.format(set_type)]['connMatrix'].T)
        edges = np.array([adj.row, adj.col]).T

        G = nx.DiGraph()
        G.add_nodes_from(range(0, np.max(edges) + 1))
        G.add_edges_from(edges)

        # G = nx.from_edgelist(edges)
        lab = np.asarray(train['{}Set'.format(set_type)]['nodeLabels']).T
        if len(lab.shape) < 2:
            lab = lab.reshape(lab.shape[0], 1)
        lab = torch.tensor(lab, dtype=torch.float)
        target = np.asarray(train['{}Set'.format(set_type)]['targets']).T
        targets = torch.tensor(np.clip(target, 0, 1), dtype=torch.long)  # convert -1 to 0

        edges, agg_matrix = nx_to_format(G, aggregation_type, sparse_matrix)

        e = len(G.edges)
        n = len(G.nodes)
        d_l = lab.shape[1]
        is_multilabel = False
        n_classes = 2
        node_labels = lab
        dset[set_type] = Dataset(
            "subgraph_{}".format(set_type),
            n,
            e,
            d_l,
            is_multilabel,
            n_classes,
            edges,
            agg_matrix,
            node_labels,
            targets)

    return dset


def get_karate(num_nodes_per_graph=None, aggregation_type="sum", sparse_matrix=True):
    # F = nx.read_edgelist("./data/karate/edges.txt", nodetype=int)
    G = nx.karate_club_graph()

    # edge = np.loadtxt("./data/karate/edges.txt", dtype=np.int32)   # 0-based indexing
    # edge_inv = np.flip(edge, axis=1)
    # edges = np.concatenate((edge, edge_inv))
    # G = nx.DiGraph()
    # G.add_edges_from(edges)
    G = G.to_directed()
    e = len(G.edges)
    n = len(G.nodes)
    # F = nx.Graph()
    # F.add_edges_from(G.edges)

    edges, agg_matrix = nx_to_format(G, aggregation_type, sparse_matrix=sparse_matrix)

    is_multilabel = False
    n_classes = 4

    targets = [0] * n
    # class_nodes = [[]] * n_classes # NB keeps broadcasting also at append time
    class_nodes = [[], [], [], []]
    with open("./data/karate/classes.txt") as f:
        for line in f:
            node, node_class = map(int, line.split(" "))
            targets[node] = node_class
            class_nodes[node_class].append(node)

    d_l = n
    # node_labels = torch.zeros(*[n, d_l])
    node_labels = torch.eye(n)
    targets = torch.tensor(targets, dtype=torch.long)

    idx_train = []
    idx_test = []
    for c in class_nodes:
        perm = np.random.permutation(c)
        idx_train += list(perm[:1])  # first index for training
        idx_test += list(perm[1:])  # all other indexes for testing
        # idx_train += list(perm)  # first index for training
        # idx_test += list(perm)  # all other indexes for testing

    idx_valid = torch.LongTensor(idx_train)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    return Dataset(
        "karate",
        n,
        e,
        d_l,
        is_multilabel,
        n_classes,
        edges,
        agg_matrix,
        node_labels,
        targets,
        idx_train,
        idx_valid,
        idx_test,
    )


def collate(samples):
    import dgl
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def get_dgl_minigc(aggregation_type="sum", ):
    import dgl
    from dgl.data import MiniGCDataset
    tr_set = MiniGCDataset(80, 10, 20)
    test_set = MiniGCDataset(20, 10, 20)
    data_loader = DataLoader(tr_set, batch_size=80, shuffle=True,
                             collate_fn=collate)
    dataiter = iter(data_loader)
    images, labels = dataiter.next()  # get all the dataset
    G = images.to_networkx()

    e = len(G.edges)
    n = len(G.nodes)

    edges, agg_matrix = nx_to_format(G, aggregation_type)

    print("ciao")


def get_dgl_cora(aggregation_type="sum", sparse_matrix=False):
    import dgl
    from dgl.data import CoraDataset

    tr_set = CoraDataset()
    G = tr_set.graph

    e = len(G.edges)
    n = len(G.nodes)
    d_l = tr_set.features.shape[1]
    is_multilabel = False
    n_classes = tr_set.num_labels
    node_labels = torch.tensor(tr_set.features)
    targets = torch.tensor(tr_set.labels)
    idx_train = torch.BoolTensor(tr_set.train_mask)  # in this case, there are msk => convert to boolean mask
    idx_valid = torch.BoolTensor(tr_set.val_mask)
    idx_test = torch.BoolTensor(tr_set.test_mask)
    edges, agg_matrix = nx_to_format(G, aggregation_type, sparse_matrix)

    return Dataset(
        "cora",
        n,
        e,
        d_l,
        is_multilabel,
        n_classes,
        edges,
        agg_matrix,
        node_labels,
        targets,
        idx_train,
        idx_valid,
        idx_test,
    )


def get_dgl_citation(aggregation_type="sum", dataset="pubmed"):
    import dgl
    from dgl.data import CitationGraphDataset

    tr_set = CitationGraphDataset(dataset)
    G = tr_set.graph

    e = len(G.edges)
    n = len(G.nodes)
    d_l = tr_set.features.shape[1]
    is_multilabel = False
    n_classes = tr_set.num_labels
    node_labels = torch.tensor(tr_set.features)
    targets = torch.tensor(tr_set.labels)
    idx_train = torch.BoolTensor(tr_set.train_mask)
    idx_valid = torch.BoolTensor(tr_set.val_mask)
    idx_test = torch.BoolTensor(tr_set.test_mask)
    edges, agg_matrix = nx_to_format(G, aggregation_type)

    return Dataset(
        "cora",
        n,
        e,
        d_l,
        is_multilabel,
        n_classes,
        edges,
        agg_matrix,
        node_labels,
        targets,
        idx_train,
        idx_valid,
        idx_test,
    )


def get_dgl_karate(aggregation_type="sum"):
    import dgl
    from dgl.data import KarateClub

    tr_set = KarateClub()
    G = tr_set.graph

    e = len(G.edges)
    n = len(G.nodes)
    d_l = tr_set.features.shape[1]
    is_multilabel = False
    n_classes = tr_set.num_labels
    node_labels = torch.tensor(tr_set.features)
    targets = torch.tensor(tr_set.labels)
    idx_train = torch.BoolTensor(tr_set.train_mask)
    idx_valid = torch.BoolTensor(tr_set.val_mask)
    idx_test = torch.BoolTensor(tr_set.test_mask)
    edges, agg_matrix = nx_to_format(G, aggregation_type)

    return Dataset(
        "cora",
        n,
        e,
        d_l,
        is_multilabel,
        n_classes,
        edges,
        agg_matrix,
        node_labels,
        targets,
        idx_train,
        idx_valid,
        idx_test,
    )


def from_EN_to_GNN(E, N, targets, aggregation_type, sparse_matrix=True):
    """
    :param E: # E matrix - matrix of edges : [[id_p, id_c, graph_id],...]
    :param N: # N matrix - [node_features, graph_id (to which the node belongs)]
    :return: # L matrix - list of graph targets [tar_g_1, tar_g_2, ...]
    """

    N_full = N
    E_full = E
    N = N[:, :-1]  # avoid graph_id
    e = E[:, :2]  # take only first tow columns => id_p, id_c

    # creating input for gnn => [id_p, id_c, label_p, label_c]

    # creating arcnode matrix, but transposed
    """
    1 1 0 0 0 0 0 
    0 0 1 1 0 0 0
    0 0 0 0 1 1 1    

    """  # for the indices where to insert the ones, stack the id_p and the column id (single 1 for column)
    G = nx.DiGraph()
    G.add_nodes_from(range(0, np.max(e) + 1))
    G.add_edges_from(e)
    edges, agg_matrix = nx_to_format(G, aggregation_type, sparse_matrix)

    # get the number of graphs => from the graph_id
    num_graphs = int(max(N_full[:, -1]) + 1)
    # get all graph_ids
    g_ids = N_full[:, -1]
    g_ids = g_ids.astype(np.int32)

    # creating graphnode matrix => create identity matrix get row corresponding to id of the graph
    # graphnode = np.take(np.eye(num_graphs), g_ids, axis=0).T
    # substitued with same code as before
    if sparse_matrix:
        unique, counts = np.unique(g_ids, return_counts=True)
        values_matrix = np.ones([len(g_ids)]).astype(np.float32)
        if aggregation_type == "degreenorm":
            values_matrix_normalized = values_matrix[g_ids] / counts[g_ids]
        else:
            values_matrix_normalized = values_matrix
        # graphnode = SparseMatrix(indices=np.stack((g_ids, np.arange(len(g_ids))), axis=1),
        #                          values=np.ones([len(g_ids)]).astype(np.float32),
        #                          dense_shape=[num_graphs, len(N)])

        agg_matrix_i = torch.LongTensor([g_ids, list(range(len(g_ids)))])
        agg_matrix_v = torch.FloatTensor(values_matrix_normalized)
        graphnode = torch.sparse.FloatTensor(agg_matrix_i, agg_matrix_v, torch.Size([num_graphs, len(N)]))
    else:
        graphnode = torch.tensor(np.take(np.eye(num_graphs), g_ids, axis=0).T)
    # print(graphnode.shape)

    e = E_full.shape[0]
    n = N_full.shape[0]
    d_l = N.shape[1]
    is_multilabel = False
    n_classes = (np.max(targets).astype(np.int) + 1)
    node_labels = torch.FloatTensor(N)
    targets = torch.tensor(targets, dtype=torch.long)
    return Dataset(
        "name",
        n,
        e,
        d_l,
        is_multilabel,
        n_classes,
        edges,
        agg_matrix,
        node_labels,
        targets,
        graph_node=graphnode
    )



def old_load_karate(path="data/karate/"):
    """Load karate club dataset"""
    print('Loading karate club dataset...')
    import random
    import scipy.sparse as sp

    edges = np.loadtxt("{}edges.txt".format(path), dtype=np.int32)  # 0-based indexing

    # edge_inv = np.flip(edges, axis=1) # add also archs in opposite direction
    # edges = np.concatenate((edges, edge_inv))
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]  # reorder list of edges also by second column
    features = sp.eye(np.max(edges+1), dtype=np.float).tocsr()

    idx_labels = np.loadtxt("{}classes.txt".format(path), dtype=np.float32)
    idx_labels = idx_labels[idx_labels[:, 0].argsort()]

    labels = idx_labels[:, 1]
    #labels = np.eye(max(idx_labels[:, 1])+1, dtype=np.int32)[idx_labels[:, 1]]  # one-hot encoding of labels

    E = np.concatenate((edges, np.zeros((len(edges), 1), dtype=np.int32)), axis=1)
    N = np.concatenate((features.toarray(), np.zeros((features.shape[0], 1), dtype=np.int32)), axis=1)

    mask_train = np.zeros(shape=(34,), dtype=np.float32)

    id_0, id_4, id_5, id_12 = random.choices(np.argwhere(labels == 0), k=4)
    id_1, id_6, id_7, id_13 = random.choices(np.argwhere(labels == 1), k=4)
    id_2, id_8, id_9, id_14 = random.choices(np.argwhere(labels == 2), k=4)
    id_3, id_10, id_11, id_15 = random.choices(np.argwhere(labels == 3), k=4)

    mask_train[id_0] = 1.  # class 1
    mask_train[id_1] = 1.  # class 2
    mask_train[id_2] = 1.  # class 0
    mask_train[id_3] = 1.  # class 3

    mask_test = 1. - mask_train

    return E, N, labels,  torch.BoolTensor(mask_train),  torch.BoolTensor(mask_test)