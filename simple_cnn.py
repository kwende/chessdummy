import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self)

        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=2,padding=2)
        self.cnn2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=2,padding=2)
        self.cnn3 = nn.Conv2d(in_channels=64,out_channels=256,kernel_size=2,padding=2)
        self.fc1 = nn.Linear(256 * 2, 100)
        self.fc2 = nn.Linear(100, 2) #??

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

