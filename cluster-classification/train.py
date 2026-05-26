import logging

from torch import nn

def train_loop(device, dataloader, model, loss_fn, optimizer):
    """Evaluates the capabilities of the current model.

    Arguments:
    - `device`: The device this model will be training on
    - `dataloader`: The data source to test the model in
    - `model`: The model being tested
    - `loss_fn`: The function used to evaluate the model.
    - `optimizer`: The method used to optimize the model at each step.

    Returns:
    - A ratio of correctly-classified clusters to all clusters considered.
    - A formatted string for user display messages.
    """

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
