'''
train.py

This file contains the code used to train the model.
'''

import logging

from torch import nn

def train_loop(device, dataloader, model, loss_fn, optimizer):
    logging.debug("Training step")
    # size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    model.to(device)

    # Loop through each batch of data from the dataloader.
    for batch_num, (data, labels) in enumerate(dataloader):
        logging.debug(f"Batch num {batch_num::>5d}")
        # Move data to GPU
        data = data.to(device)
        labels = labels.to(device)
        # Run the model and calculate loss for all items in the batch
        pred = model(data)
        loss = loss_fn(pred, labels)

        # Run backprop and the optimizer to train the model for the next run 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
