'''
model.py

This file contains the definition for the model used in training.
'''

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # Define the network once data has been acquired
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
