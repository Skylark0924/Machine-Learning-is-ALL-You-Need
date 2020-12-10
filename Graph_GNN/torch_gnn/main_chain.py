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
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100000, metavar='N',
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
    cfg.state_transition_hidden_dims = [15]
    cfg.output_function_hidden_dims = [5]
    cfg.state_dim = 10
    cfg.max_iterations = 50
    cfg.convergence_threshold = 0.001
    cfg.graph_based = False
    cfg.log_interval = 10
    cfg.task_type = "semisupervised"

    cfg.lrw = 0.01

    # model creation
    model = SemiSupGNNWrapper(cfg)
    # dataset creation
   # dset = dataloader.get_karate(aggregation_type="sum", sparse_matrix=True)  # generate the dataset
    dset = dataloader.get_twochainsSSE(aggregation_type="sum", percentage=0.1, sparse_matrix=True)  # generate the dataset
    model(dset)  # dataset initalization into the GNN

    # training code
    for epoch in range(1, args.epochs + 1):
        model.train_step(epoch)

        if epoch % 10 == 0:
            model.test_step(epoch)
    # model.test_step()

    # if args.save_model:
    #     torch.save(model.gnn.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
