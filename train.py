import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        