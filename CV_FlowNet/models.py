import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np
from networks import FlowNetE, FlowNetER, FlowNetERM

class FlowNetE(FlowNetE.FlowNetEncoder):
    def __init__(self, args, div_flow=20):
        super(FlowNetE,self).__init__(args, input_channels=6, div_flow=div_flow)
        print('div_flow', div_flow)

class FlowNetER(FlowNetER.FlowNetEncoderRefine):
    def __init__(self, args, div_flow=20):
        super(FlowNetER,self).__init__(args, input_channels=6, div_flow=div_flow)
        print('div_flow', div_flow)

class FlowNetERM(FlowNetERM.FlowNetEncoderRefineMultiscale):
    def __init__(self, args, div_flow=20):
        super(FlowNetERM,self).__init__(args, input_channels=6, div_flow=div_flow)
        print('div_flow', div_flow)

