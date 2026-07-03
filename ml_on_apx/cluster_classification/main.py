"""Implements the testing-training loop for a cluster classification model."""

import os

import numpy as np
import torch
from torch import nn

from ml_on_apx.cleverlogger import CleverLogger
from ml_on_apx.cluster_classification.model import Model
from ml_on_apx.cluster_classification.test import test_loop
from ml_on_apx.cluster_classification.train import train_loop


def main() -> None:
    """Trains a cluster classification model.

    Raises:
        NotImplementedError: We're not done rebuilding it yet.

    """
    raise NotImplementedError("Must be redone")
    logger = CleverLogger(__name__)
    # After how many epochs should a checkpoint be made?
    # CHECKPOINT_RATE = 10
    # If accuracy growth falls below this value, stop training early
    # GROWTH_THRESHOLD = [0.0001 for _ in range(3)]
    # If accuracy reaches this this threshold, then stop training.
    # STOP_THRESHOLD = 0.85
    # Learning rate of the model.
    # LEARNING_RATE = 1e-4
    # How many data points to analyze in a batch.
    # BATCH_SIZE = 1

    temp = 2

    # Instantiate a Model object:
    model = Model()

    # Set device
    current_device = torch.accelerator.current_accelerator(check_available=True)
    if current_device is not None:
        device = current_device
    else:
        device = torch.device("cpu")

    logger.log_notice(f"Using {device.type} device")

    # Collect data
    logger.log_start_major_process("load_training_data")
    training_data, weights = temp, temp
    # get_data(
    #     DatasourceType.TRAINING, current_device, BATCH_SIZE
    # )
    logger.log_end_major_process("load_training_data")

    logger.log_start_major_process("load_testing_data")
    testing_data, _ = (
        temp,
        temp,
    )  # get_data(DatasourceType.TESTING, current_device, BATCH_SIZE)
    logger.log_end_major_process("load_testing_data")

    # Set loss function
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    loss_fn.to(device)

    # Stochastic Gradient Descent
    optimizer = torch.optim.SGD(model.parameters(), lr=temp)  # Learn rate

    last_acc = 0.0
    growth = [np.inf for _ in range(len(temp))]  # Growth thresh
    sentinal = True
    epoch = 0
    # Epoch loop
    logger.log_start_major_process("train_test_loop")
    logger.log_preloop("epoch_while_loop")
    while sentinal:
        logger.log_iteration_head(Epoch=epoch)
        logger.log_start_minor_process("training")
        # Run through the training data once
        train_loop(device, training_data, model, loss_fn, optimizer)
        logger.log_end_minor_process("training")

        logger.log_start_minor_process("testing")
        # Run through the testing data once and evaluate the model's accuracy
        acc, string = test_loop(device, testing_data, model, loss_fn)
        logger.log_end_minor_process("testing")

        # If the epoch is a checpoint epoch,
        if epoch % temp == 0:  # checkpoint rate
            logger.log_notice(
                "Checkpoint reached",
                checkpoint=(epoch // temp),  # chkpt rate
                status=string,
            )
            # save the model as a checkpoint
            torch.save(
                model,
                # checkpoint rate
                f"./models/classification/checkpoint-{(epoch // temp):>05d}.pth",
            )
            if acc > temp:  # stop thresh
                # If the accuracy is hight enough, exit training
                logger.log_notice(f"Accuracy threshold reached: {acc}")
                sentinal = False

            growth.pop()
            growth.append(acc - last_acc)
            if np.all(np.array(growth) < temp):  # growth thresh
                # If the accuacy has not grown appreciably since last test, exit
                # training
                logger.log_notice(f"Accuracy growth limit reached: {acc - last_acc}")
                sentinal = False

            last_acc = acc
        epoch = epoch + 1
        logger.log_iteration_tail()
    logger.log_postloop("epoch_while_loop")
    logger.log_end_major_process("train_test_loop")

    # Softlink the last checkpoint
    os.symlink(
        f"checkpoint-{(epoch // temp):>05d}-classification.pth",  # checkpoint rate
        "current-classification.pth",
    )
