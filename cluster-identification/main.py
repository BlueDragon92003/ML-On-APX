'''
main.py

This python script is executed to train and test the 
'''

import logging
import os

import torch
from torch import nn

from model import NeuralNetwork
from constants import *

from model import Model
from test import test_loop
from train import train_loop
from data import get_data, Type

# After how many epochs should a checkpoint be made?
CHECKPOINT_RATE = 10
# If accuracy growth falls below this value, stop training early
GROWTH_THRESHOLD = 0.001
# If accuracy reaches this this threshold, then stop training.
STOP_THRESHOLD = 0.95
# Learning rate of the model.
LEARNING_RATE = 1e-4
# How many data points to analyze in a batch.
BATCH_SIZE = 1

logging.basicConfig(filename='./logs/identification-latest.log', level=logging.INFO)

# Set device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
logging.info(f"Using {device} device")

# Set loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# Collect data
training_data = get_data(Type.TRAINING, BATCH_SIZE)

# Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Epoch loop
while(sentinal):
    # Unsupervised learning aproach may be best, so I'm leaving this empty for
    # the moment. We shall see what works best. 
    pass

# Softlink the last checkpoint
os.symlink( f"checkpoint-{(epoch // CHECKPOINT_RATE):>05d}-identification.pth", "current-identification.pth" )

