"""Implements the testing-training loop for a cluster classification model."""

import random
from datetime import datetime
from pathlib import Path
from typing import cast

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
from ml_on_apx.labelling import Labels
from ml_on_apx.logging import log_call
from ml_on_apx.model_management.model_info import ModelInfo, ModelTestInfo
from ml_on_apx.model_management.model_manager import ModelManager
from ml_on_apx.model_management.stop_functions import StopFunction
from ml_on_apx.modes import Mode

_MAIN = "main" @ _CLASS


def get_device() -> torch.device:
    """Get the ML device."""
    current_device = torch.accelerator.current_accelerator(check_available=True)
    if current_device is not None:
        device = current_device
    else:
        device = torch.device("cpu")
    return device


def get_weights(
    data: ClusterClassificationDataset,
    labels: Labels,
) -> torch.Tensor:
    """Get the weights of the dataset."""
    return torch.tensor(
        [
            (data.size_per_label[i] if i in data.size_per_label.keys() else 1)
            for i in range(len(labels))
        ],
        dtype=torch.float,
    )


@log_call(action_type="train" > _MAIN, include_result=False)
def train(data_dir: Path, model_dir: Path) -> None:  # noqa: PLR0915
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

        training_data = cast(ClusterClassificationDataset, training_data)
        testing_data = cast(ClusterClassificationDataset, testing_data)
        manager.training_job = None

        acc_ls: list[float] = []
        loss_ls: list[float] = []
        epoch_ls: list[float] = []

        labels = group.get_labels(manager)
        training_weights = get_weights(training_data, labels)
        testing_weights = get_weights(testing_data, labels)

        training_data_loader = DataLoader(training_data)
        testing_data_loader = DataLoader(testing_data)

        # Set device
        device = get_device()

        # Set loss function
        training_loss_fn = nn.CrossEntropyLoss(weight=training_weights)
        training_loss_fn.to(device)
        testing_loss_fn = nn.CrossEntropyLoss(weight=testing_weights)
        testing_loss_fn.to(device)

        # Stochastic Gradient Descent & Learn rate
        optimizer = torch.optim.SGD(model.parameters(), lr=job.learning_rate)

        sentinal = True
        epoch = 0
        # Epoch loop
        while sentinal:
            # Run through the training data once
            train_loop(device, training_data_loader, model, training_loss_fn, optimizer)

            # Run through the testing data once and evaluate the model's accuracy
            acc, loss = test_loop(device, testing_data_loader, model, testing_loss_fn)

            # If the epoch is a checpoint epoch,
            if epoch % job.checkpoint_rate == 0:  # checkpoint rate
                log_message("checkpoint" > _MAIN, acc=acc, loss=loss, epoch=epoch)

                acc_ls.append(acc)
                loss_ls.append(loss)
                epoch_ls.append(float(epoch))

                acc_ls = acc_ls[-(job.lookback_distance) :]
                loss_ls = loss_ls[-(job.lookback_distance) :]
                epoch_ls = epoch_ls[-(job.lookback_distance) :]

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
                info.add_testing_information(ModelTestInfo(datetime.now(), acc, loss))

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


@log_call(action_type="test" > _MAIN, include_result=False)
def test(data_dir: Path, model_dir: Path) -> None:
    """Test a cluster classification model."""
    device = get_device()

    with ModelManager(model_dir, Mode.Classification) as manager:
        job = manager.testing_job
        if job is None:
            print("No job set!")
            return
        model_info = manager.get_model_info(*job.target)
        model = manager.get_model(*job.target)

        with DatasetManager(
            data_dir, Mode.Classification, Mode.Classification.dataset_class
        ) as data_manager:
            testing_data = data_manager.get_dataset(job.dataset)

        testing_data = cast(ClusterClassificationDataset, testing_data)
        labels = manager.get_group_info(job.target[0]).get_labels(manager)
        loss_fn = nn.CrossEntropyLoss(weight=get_weights(testing_data, labels))
        loss_fn.to(device)
        testing_data_loader = DataLoader(testing_data)

        acc, loss = test_loop(device, testing_data_loader, model, loss_fn)
        model_info.add_testing_information(
            ModelTestInfo(datetime.now(), acc, loss, run_by_user=True)
        )
