"""Trains a cluster classification model."""

import torch
from torch import nn

from ml_on_apx.cluster_classification import _CLASS
from ml_on_apx.logging import log_call

_TRAIN = "train" @ _CLASS


@log_call(action_type="main" @ _TRAIN)
def train_loop(
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Improve the capabilities of the model.

    Arguments:
        device (torch.device): The device this model will be training on
        dataloader (torch.utils.data.DataLoader): The data source to test the model in
        model (nn.Module): The model being tested
        loss_fn (nn.Modul): The function used to evaluate the model.
        optimizer (torch.optim.Optimizer): The method used to optimize the model at each
            step.

    Returns:
    - A ratio of correctly-classified clusters to all clusters considered.
    - A formatted string for user display messages.

    """
    # size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and
    #       dropout layers
    model.train()
    model.to(device)

    for _batch_num, batch_items in enumerate(dataloader):
        # Move data to GPU
        data = batch_items[:, :-1].to(device)
        labels = batch_items[:, -1].to(device).type(torch.long)
        # Run the model and calculate loss for all items in the batch
        pred = model(data)
        loss = loss_fn(pred, labels)

        # Run backprop and the optimizer to train the model for the next run
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
