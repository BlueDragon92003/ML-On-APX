from typing import List
from ml_on_apx.dataset_management.dataset_info import DatasetInfo
from datetime import datetime, date
from ml_on_apx.labelling import Labels


class ModelTestInfo:
    """Stores data corresponding to a single model test."""

    def __init__(
        self,
        test_time: datetime,
        accuracy: float,
        average_loss: float,
        run_by_user: bool = False,
    ):
        self._test_time: datetime = test_time
        self._run_by_user: bool = run_by_user
        self._accuracy: float = accuracy
        self._average_loss: float = average_loss

        # TODO Properties for all of the above


class ModelInfo:
    """Stores training & testing info about a model."""

    def __init__(
        self,
        start_date: date,
        fork_time: datetime,
        labels: Labels,
        training_datasets: List[DatasetInfo],
    ):
        # Training information
        self._training_start_date: date = start_date
        self._model_fork_time: datetime = fork_time
        self._labels: Labels = labels
        self._training_datasets: List[DatasetInfo] = training_datasets
        # Testing information
        self._testing_information: List[ModelTestInfo] = list()

        # TODO Properties for all of the above

    def add_testing_information(self, new_test: ModelTestInfo):
        self._testing_information.append(new_test)
