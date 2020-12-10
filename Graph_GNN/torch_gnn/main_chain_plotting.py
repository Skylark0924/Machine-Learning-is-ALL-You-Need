import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import utils
import dataloader

from gnn_wrapper import GNNWrapper, SemiSupGNNWrapper


#
# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda_dev', type=int, default=0,
                        help='select specific CUDA device for training')
    parser.add_argument('--n_gpu_use', type=int, default=1,
                        help='select number of CUDA device for training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='logging training status cadency')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='For logging the model in tensorboard')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        args.n_gpu_use = 0

    device = utils.prepare_device(n_gpu_use=args.n_gpu_use, gpu_id=args.cuda_dev)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # torch.manual_seed(args.seed)
    # # fix random seeds for reproducibility
    # SEED = 123
    # torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(SEED)

    # configugations
    cfg = GNNWrapper.Config()
    cfg.use_cuda = use_cuda
    cfg.device = device

    cfg.log_interval = args.log_interval
    cfg.tensorboard = args.tensorboard

    # cfg.batch_size = args.batch_size
    # cfg.test_batch_size = args.test_batch_size
    # cfg.momentum = args.momentum

    cfg.dataset_path = './data'
    cfg.epochs = args.epochs
    cfg.lrw = args.lr
    cfg.activation = nn.Tanh()
    cfg.state_transition_hidden_dims = [4]
    cfg.output_function_hidden_dims = []
    cfg.state_dim = 2
    cfg.max_iterations = 50
    cfg.convergence_threshold = 0.001
    cfg.graph_based = False
    cfg.log_interval = 10
    cfg.task_type = "semisupervised"

    cfg.lrw = 0.01

    # model creation
    model = SemiSupGNNWrapper(cfg)
    # dataset creation
    dset = dataloader.get_karate(aggregation_type="sum", sparse_matrix=True)  # generate the dataset
    #dset = dataloader.get_twochainsSSE(aggregation_type="sum", percentage=0.1, sparse_matrix=True)  # generate the dataset
    model(dset)  # dataset initalization into the GNN

    # training code

    # plotting utilities
    all_states = []
    all_outs = []
    for epoch in range(1, args.epochs + 1):
        out = model.train_step(epoch)
        all_states.append(model.gnn.converged_states.detach().to("cpu"))
        all_outs.append(out.detach().to("cpu"))

        if epoch % 10 == 0:
            model.test_step(epoch)
    # model.test_step()

    # if args.save_model:
    #     torch.save(model.gnn.state_dict(), "mnist_cnn.pt")

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import networkx as nx
    nx_G = nx.karate_club_graph().to_directed()

    def draw(i):
        clscolor = ['#FF0000', '#0000FF', '#FF00FF', '#00FF00']
        pos = {}
        colors = []
        for v in range(34):
            pos[v] = all_states[i][v].numpy()
            cls = all_outs[i][v].argmax(axis=-1)
            # colors.append(clscolor[cls])
            # print(clscolor[targets[v]])
            colors.append(clscolor[dset.targets[v]])
        ax.cla()
        ax.axis('off')
        ax.set_title('Epoch: %d' % i)
        #     node_sha = ["o" for i in range(34)]
        #     for j in idx_train:
        #         node_sha[j] = "s"
        node_sizes = np.full((34), 200)
        node_sizes[dset.idx_train.detach().to("cpu").numpy()] = 350
        nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
                         with_labels=True, node_size=node_sizes, ax=ax)

    #     nx.draw_networkx(nx_G.to_undirected().subgraph(idx_train), pos, node_color=[colors[k] for k in idx_train], node_shape='s',
    #             with_labels=True, node_size=300, ax=ax)

    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()
    draw(0)  # draw the prediction of the first epoch
    plt.close()

    ani = animation.FuncAnimation(fig, draw, frames=len(all_states), interval=200)
    ani.save('learning.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

if __name__ == '__main__':
    main()
