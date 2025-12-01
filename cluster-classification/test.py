'''
test.py

This file contains the code used to test the model.
'''

import logging
import torch

def test_loop(device, dataloader, model, loss_fn):
    """
    Test the current training state of the model and report this status to both
    the logging system and program control to provide an estimation of progress
    to the observer and determine whether or not training should stop.
    """

    logging.debug("Testing step")
    # Set the model to evaluation mode
    model.eval()
    model.to(device)
    
    # Set up tracking values to evaluate the success of the model
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return correct, f"\tAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}"
