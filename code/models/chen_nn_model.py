import pandas as pd
import numpy as np 
import torch
import torch.nn as nn

##### Torch Models #####
class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()
        layers = [nn.Linear(input_size,layer_sizes[0]), nn.ReLU(), nn.Dropout(p=0.01)]
        for i in range(len(layer_sizes)-1):
            hidden_layer = [nn.Linear(layer_sizes[i],layer_sizes[i+1]), nn.ReLU(),  nn.Dropout(p=0.01)]
            layers += hidden_layer

        layers += [nn.Linear(layer_sizes[-1],1),nn.ReLU()]
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        u = self.linear_relu_stack(x)
        return u