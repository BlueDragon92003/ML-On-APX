"""Implements the testing-training loop for a cluster classification model."""

import random
from datetime import datetime
from pathlib import Path

import torch
from eliot import log_message
from torch import nn
from torch.utils.data import DataLoader

from ml_on_apx.cluster_classification import _CLASS
from ml_on_apx.cluster_classification.cluster_classification_dataset import (
    ClusterClassificationDataset,
)
from ml_on_apx.cluster_classification.test import test_loop
from ml_on_apx.cluster_classification.train import train_loop
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from ml_on_apx.logging import log_call
from ml_on_apx.model_management.model_info import ModelInfo, ModelTestInfo
from ml_on_apx.model_management.model_manager import ModelManager
from ml_on_apx.model_management.stop_functions import StopFunction
from ml_on_apx.modes import Mode

_MAIN = "main" @ _CLASS


@log_call(action_type="main" > _MAIN, include_result=False)
def main(data_dir: Path, model_dir: Path) -> None:  # noqa: PLR0915
    """Train a cluster classification model."""
    start_date = datetime.today()
    instance = random.randbytes(4).hex()

    with ModelManager(model_dir, Mode.Classification) as manager:
        job = manager.training_job
        if job is None:
            print("No job set!")
            return
        group = manager.get_group_info(job.group_name)
        if job.base_model_name:
            model = manager.get_model(job.group_name, job.base_model_name)
        else:
            model = group.model()

        with DatasetManager(
            data_dir, Mode.Classification, Mode.Classification.dataset_class
        ) as data_manager:
            training_data = data_manager.get_dataset(job.dataset)
            if job.testing_dataset is None:
                testing_data = training_data
            else:
                testing_data = data_manager.get_dataset(job.testing_dataset)

        assert type(training_data) is ClusterClassificationDataset
        manager.training_job = None

        acc_ls: list[float] = []
        loss_ls: list[float] = []
        epoch_ls: list[float] = []

        weights = torch.tensor(
            [
                (
                    training_data.size_per_label[i]
                    if i in training_data.size_per_label.keys()
                    else 1
                )
                for i in range(len(group.get_labels(manager)))
            ],
            dtype=torch.float,
        )

        training_data_loader = DataLoader(training_data)
        testing_data_loader = DataLoader(testing_data)

        # Set device
        current_device = torch.accelerator.current_accelerator(check_available=True)
        if current_device is not None:
            device = current_device
        else:
            device = torch.device("cpu")

        # Set loss function
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss_fn.to(device)

        # Stochastic Gradient Descent & Learn rate
        optimizer = torch.optim.SGD(model.parameters(), lr=job.learning_rate)

        sentinal = True
        epoch = 0
        # Epoch loop
        # logger.log_start_major_process("train_test_loop")
        # logger.log_preloop("epoch_while_loop")
        while sentinal:
            # logger.log_iteration_head(Epoch=epoch)
            # logger.log_start_minor_process("training")
            # Run through the training data once
            train_loop(device, training_data_loader, model, loss_fn, optimizer)
            # logger.log_end_minor_process("training")

            # logger.log_start_minor_process("testing")
            # Run through the testing data once and evaluate the model's accuracy
            acc, loss = test_loop(device, testing_data_loader, model, loss_fn)
            # logger.log_end_minor_process("testing")

            # If the epoch is a checpoint epoch,
            if epoch % job.checkpoint_rate == 0:  # checkpoint rate
                log_message("checkpoint" > _MAIN, acc=acc, loss=loss, epoch=epoch)

                acc_ls.append(acc)
                loss_ls.append(loss)
                epoch_ls.append(float(epoch))

                acc_ls = acc_ls[-(job.lookback_distance) :]

                try:
                    result = job.stop_function(
                        ACC=list(reversed(acc_ls)),
                        LOSS=list(reversed(loss_ls)),
                        EPOCH=list(reversed(epoch_ls)),
                    )
                except StopFunction.EvaluationError as e:
                    print(f"An error occurred in evaluation:\n{type(e)}\t{e.args}")
                    error = e
                    sentinal = False
                else:
                    error = None
                    sentinal = not result

                info = ModelInfo(
                    start_date=start_date,
                    fork_time=datetime.now(),
                    group=job.group_name,
                    training_dataset=job.dataset,
                    stop_function_errored=error,
                )
                info.add_testing_information(
                    ModelTestInfo(
                        datetime.now(),
                        acc,
                        loss,
                    )
                )

                (
                    manager.create_model(
                        job.group_name,
                        f"~checkpoint-{start_date.isoformat()}"
                        f"-{instance}"
                        f"-{epoch // job.checkpoint_rate}",
                        info,
                        model,
                    ),
                )

                print()

            epoch = epoch + 1
