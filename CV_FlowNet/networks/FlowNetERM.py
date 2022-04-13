import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np


class FlowNetEncoderRefineMultiscale(nn.Module):
    def __init__(self, args, input_channels=12, batchNorm=True, div_flow=20):
        super(FlowNetEncoderRefineMultiscale, self).__init__()

        self.rgb_max = args.rgb_max
        self.div_flow = div_flow  # A coefficient to obtain small output value for easy training, ignore it

        '''Implement Codes here'''
        self.batchNorm = False
        self.convL1 = self.convL(self.batchNorm, 6, 64, kernel_size=7, stride=2)
        self.convL2 = self.convL(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.convL3 = self.convL(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.convL4 = self.convL(self.batchNorm, 256, 512, kernel_size=1, stride=2)
        self.convL4_1 = self.convL(self.batchNorm, 512, 512)

        self.deconvL3 = self.deconvL(512, 256)
        self.deconvL2 = self.deconvL(514, 128)

        self.conv4 = self.conv(512)
        self.conv3 = self.conv(514)

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

    def deconvL(self, in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def conv(self, in_planes):
        return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)

    def crop_like(self, input, target):
        if input.size()[2:] == target.size()[2:]:
            return input
        else:
            return input[:, :, :target.size(2), :target.size(3)]

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

        flow4 = self.conv4(out_conv4)
        flow4_up = self.crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = self.crop_like(self.deconv5(out_conv3), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.conv3(concat3)
        flow3_up = self.crop_like(self.upsampled_flow5_to_4(flow3), out_conv2)
        out_deconv2 = self.crop_like(self.deconv4(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.conv2(concat2)

        if self.training:
            return flow2, flow3, flow4
        else:
            return flow2 * self.div_flow
