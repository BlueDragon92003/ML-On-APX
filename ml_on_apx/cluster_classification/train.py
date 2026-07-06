"""Trains a cluster classification model."""

import torch
from torch import nn


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
    # logger.log_enter_function(
    #     "train_loop_fn",
    #     device=device,
    #     dataloader=dataloader,
    #     model=model,
    #     loss_fn=loss_fn,
    #     optimizer=optimizer,
    # )
    # size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and
    #       dropout layers
    model.train()
    model.to(device)

    # Loop through each batch of data from the dataloader.
    # logger.log_preloop("training_for_loop")
    #
    for batch_num, batch_items in enumerate(dataloader):
        # logger.log_iteration_head(batch_num=batch_num)
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
        # logger.log_iteration_tail()
    # logger.log_postloop("training_for_loop")
    # logger.log_exit_function("train_loop_fn")
