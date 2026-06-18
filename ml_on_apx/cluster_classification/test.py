from typing import Tuple

import torch

from cleverlogger import CleverLogger

logger = CleverLogger(__name__)

logger.log_start_load_module()


def test_loop(
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
) -> Tuple[float, str]:
    """Evaluates the capabilities of the current model.

    Arguments:
    - `device`: The device this model will be training on
    - `dataloader`: The data source to test the model in
    - `model`: The model being tested
    - `loss_fn`: The function used to evaluate the model.

    Returns:
    - A ratio of correctly-classified clusters to all clusters considered.
    - A formatted string for user display messages.
    """

    logger.log_enter_function(
        "test_loop_fn",
        device=device,
        dataloader=dataloader,
        model=model,
        loss_fn=loss_fn,
    )
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
        logger.log_preloop("testing_for_loop")
        for cluster in dataloader:
            logger.log_iteration_head(cluster=cluster)
            # Move data to GPU
            data = cluster[:, :-1].to(device)
            label = cluster[:, -1].to(device).type(torch.long)
            pred = model(data)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            logger.log_iteration_tail()
        logger.log_postloop("testing_for_loop")

    test_loss /= num_batches
    correct /= size
    outstring = f"\tAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}"
    logger.log_function_exit_type("return", retval=[correct, outstring])
    logger.log_exit_function("test_loop_fn")
    return correct, outstring


logger.log_end_load_module()
