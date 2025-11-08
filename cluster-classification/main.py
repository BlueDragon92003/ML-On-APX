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
from data import get_data, DatasourceType

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

logging.basicConfig(filename='./logs/classification-latest.log', level=logging.INFO)

# Instantiate a Model object:
model = Model()

# Set device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
logging.info(f"Using {device} device")

# Set loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# Collect data
training_data = get_data(DatasourceType.TRAINING, BATCH_SIZE)
testing_data  = get_data(DatasourceType.TESTING, BATCH_SIZE)

# Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

last_acc = 0
sentinal = True
epoch = 0
# Epoch loop
while(sentinal):
    logging.info(f"Epoch {epoch}")
    logging.debug("Entering training loop")
    # Run through the training data once
    train_loop(training_data, model, loss_fn, optimizer)
    logging.debug("Entering testing loop")
    # Run through the testing data once and evaluate the model's accuracy
    acc, string = test_loop(testing_data, model, loss_fn)
    # If the epoch is a checpoint epoch,
    if (epoch % CHECKPOINT_RATE == 0):
        logging.debug("Checkpoint")
        logging.debug(string)
        # save the model as a checkpoint
        torch.save(model, f"checkpoint-{(epoch // CHECKPOINT_RATE):>05d}-classification.pth")
        if (acc > STOP_THRESHOLD):
            # If the accuracy is hight enough, exit training
            logging.info("Accuracy threshold reached.")
            sentinal = False
        if (acc - last_acc < GROWTH_THRESHOLD):
            # If the accuacy has not grown appreciably since last test, exit
            # training
            logging.info("Accuracy growth limit reached.")
            sentinal = False
        last_acc = acc
    epoch = epoch + 1

# Softlink the last checkpoint
os.symlink( f"checkpoint-{(epoch // CHECKPOINT_RATE):>05d}-classification.pth", "current-classification.pth" )

