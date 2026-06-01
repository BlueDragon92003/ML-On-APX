import torch
from torch import nn

from cluster_classification.classification_logger import CleverLogger

logger = CleverLogger(__name__)

logger.log_start_load_module()


def train_loop(
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
):
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

    logger.log_enter_function(
        "train_loop_fn",
        device=device,
        dataloader=dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )
    # size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and
    #       dropout layers
    model.train()
    model.to(device)

    # Loop through each batch of data from the dataloader.
    logger.log_open_control_flow("training_for_loop")
    for batch_num, (data, labels) in enumerate(dataloader):
        logger.log_open_control_flow("iteration", batch_num=batch_num)
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
        logger.log_close_control_flow("Iteration")
    logger.log_close_control_flow("training_for_loop")
    logger.log_exit_function("train_loop_fn")


logger.log_end_load_module()
