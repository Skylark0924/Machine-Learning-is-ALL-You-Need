import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        # You cannot directly use any loss functions from torch.nn or torch.nn.functional, other modules are free to use.

    def softmax(self, x):
        exps = torch.exp(x)
        return exps / torch.sum(exps)

    def forward(self, x, y, **kwargs):
        m = y.shape[0]
        p = self.softmax(x)
        # We use multidimensional array indexing to extract
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        log_likelihood = -torch.log(p[range(m), y])
        loss = torch.sum(log_likelihood) / m
        return loss



