+'''
model.py

This file contains the definition for the model used in training.
'''

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Other layers to try: Dropout and batch normalization,
        # if they make any sense. It's a small model though, so likely not
        self.stack = nn.Sequential(
            # Eta, Phi, HCAL energies, HCAL deposition, Cluster Energy
            nn.Linear(2+9+9+1, 16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8, 2),
            # Hadronic Certainty, Electromagnetic certainty
        )

    def forward(self, x):
        certainties = self.stack(x)
        return certainties

