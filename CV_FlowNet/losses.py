import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EPELoss(nn.Module):
    def __init__(self, args, div_flow=0.05):
        super(EPELoss, self).__init__()
        self.div_flow = div_flow
        self.loss_labels = ['EPE'],

    def forward(self, output, target):
        epevalue = 0
        target = self.div_flow * target
        assert output.shape == target.shape, (output.shape, target.shape)
        ''' Implement the EPE loss here'''
        ''''''
        EPE_map = torch.norm(target - output, 2, 1)
        epevalue = EPE_map.mean()
        return [epevalue]


class MultiscaleLoss(nn.Module):
    def __init__(self, args):
        super(MultiscaleLoss, self).__init__()

        self.args = args
        self.div_flow = 0.05
        self.loss_labels = ['Multiscale'],
        ''' Implement the MultiScale loss here'''
        l_weight = 0.32
        self.startScale = 4
        self.numScales = 3
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.multiScales = [nn.AvgPool2d(self.startScale * (2 ** scale), self.startScale * (2 ** scale)) for scale in
                            range(self.numScales)]

    def EPE(self, input_flow, target_flow):
        return torch.norm(target_flow - input_flow, p=2, dim=1).mean()

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0
        target = self.div_flow * target
        for i, output_ in enumerate(output):
            target_ = F.interpolate(target, output_.shape[2:], mode='bilinear', align_corners=False)
            assert output_.shape == target_.shape, (output_.shape, target_.shape)
            ''' Implement the MultiScale loss here'''
            target_ = self.multiScales[i](target)
            epevalue += self.loss_weights[i] * self.EPE(output_, target_)
        return [epevalue]
