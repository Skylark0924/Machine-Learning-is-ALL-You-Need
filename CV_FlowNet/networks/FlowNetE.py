import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np


class FlowNetEncoder(nn.Module):
    def __init__(self, args, input_channels=6, div_flow=20):
        super(FlowNetEncoder, self).__init__()

        self.rgb_max = args.rgb_max
        self.div_flow = div_flow  # A coefficient to obtain small output value for easy training, ignore it
        '''Implement Codes here'''
        self.batchNorm = False
        self.convL1 = self.convL(self.batchNorm, 6, 64, kernel_size=7, stride=2)
        self.convL2 = self.convL(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.convL3 = self.convL(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.convL4 = self.convL(self.batchNorm, 256, 512, kernel_size=1, stride=2)
        self.convL4_1 = self.convL(self.batchNorm, 512, 512)
        self.out = nn.Conv2d(512, 128, kernel_size=1, stride=2, padding=0, bias=True)

    def convL(self, batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )

    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)
        ##
        '''Implement Codes here'''
        out_conv2 = self.convL2(self.convL1(x))
        out_conv3 = self.convL3(out_conv2)
        out_conv4 = self.convL4_1(self.convL4(out_conv3))

        flow4 = nn.functional.interpolate(self.out(out_conv4), scale_factor=2)

        if self.training:
            return flow4
        else:
            return flow4 * self.div_flow
