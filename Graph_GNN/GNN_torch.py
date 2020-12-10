import torch
import utils
import sys
sys.path.append('/home/skylark/Github/Machine-Learning-Basic-Codes/Graph_GNN')
import torch_gnn.dataloader
from torch_gnn.gnn_wrapper import GNNWrapper, SemiSupGNNWrapper


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