"""Tests a model's training progess."""

import torch

from ml_on_apx.cluster_classification import _CLASS
from ml_on_apx.logging import log_call

_TEST = "test" @ _CLASS


@log_call(action_type="main" @ _TEST)
def test_loop(
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
) -> tuple[float, float]:
    """Evaluate the capabilities of the model.

    Arguments:
        device (torch.device): The device this model will be training on
        dataloader (torch.utils.data.DataLoader): The data source to test the model in
        model (torch.nn.Module): The model being tested
        loss_fn (torch.nn.Model): The function used to evaluate the model.

    Returns:
        tuple[float, str]: A ratio of correctly-classified clusters to all clusters
            considered, and a formatted string for user display messages.

    """
    # logger.log_enter_function(
    #     "test_loop_fn",
    #     device=device,
    #     dataloader=dataloader,
    #     model=model,
    #     loss_fn=loss_fn,
    # )
    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    # Set up tracking values to evaluate the success of the model
    size = len(dataloader.dataset)  # ty: ignore[invalid-argument-type]
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0.0

    # Evaluating the model with torch.no_grad() ensures that no gradients are
    #       computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage
    #       for tensors with requires_grad=True

    with torch.no_grad():
        # logger.log_preloop("testing_for_loop")
        for cluster in dataloader:
            # logger.log_iteration_head(cluster=cluster)
            # Move data to GPU
            data = cluster[:, :-1].to(device)
            label = cluster[:, -1].to(device).type(torch.long)
            pred = model(data)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
        #     logger.log_iteration_tail()
        # logger.log_postloop("testing_for_loop")

    test_loss /= num_batches
    correct /= size
    return correct, test_loss
