import torch
import torchvision

import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

import numpy as np
import sys
import os

def print_model_parm_flops(model,multiply_adds = False):
    list_conv=[]
    list_conv_params=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)
        list_conv_params.append(params)

    list_linear=[] 
    list_linear_params=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
        list_linear_params.append(flops)

    list_bn=[] 
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu=[] 
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()

        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.LeakyReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

        
    foo(model)
    # input = Variable(torch.rand(3, 32, 32).unsqueeze(0), requires_grad = True)
    input = Variable(torch.rand(3, 2, 192, 256).unsqueeze(0), requires_grad = False)
    out = model(input)


    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    total_params=(sum(list_conv_params)+sum(list_linear_params))

    print('conv: %.2fM' % (sum(list_conv)/1e6))
    print('linear: %.2fM' % (sum(list_linear)/1e6))
    print('bn: %.2fM' % (sum(list_bn)/1e6))
    print('relu: %.2fM' % (sum(list_relu)/1e6))
    print('pool: %.2fM' % (sum(list_pooling)/1e6))
    print('  + Number of FLOPs: %.2fM' % (total_flops / 1e6))

    print("conv %.2fM"%(sum(list_conv_params)/1e6))
    print("linear: %.2fM" %(sum(list_linear_params)/1e6))
    print("total_params %.2fM"%(total_params/1e6))
