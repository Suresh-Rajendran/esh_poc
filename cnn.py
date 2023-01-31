import torch
import torch.nn as nn
from activation import *

class CNet(nn.Module):   
    def __init__(self, num_classes = 10, act = 'esh'):
        super(CNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            Activation(act),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            Activation(act),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 8 * 8, num_classes)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class NNet(nn.Module):
    def __init__(self, act = 'esh', depth=15):
        super(NNet, self).__init__()
        self.act = act
        self.depth = depth
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32, 512),
            Activation(self.act),
            nn.Linear(512, 512),
            Activation(self.act),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def simpleCNet(num_classes = 10, act = 'esh'):
    return CNet(n_classes=num_classes, act = act)

def simpleNNet(num_classes = 10, act = 'esh', depth=15):
    return NNet(n_classes=num_classes, act = act)
