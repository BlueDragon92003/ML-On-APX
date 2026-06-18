from ml_on_apx.model_management.stop_functions import StopFunction
from typing import List, Callable
from ml_on_apx.dataset_management.dataset_info import DatasetInfo


class TrainingJob:
    """Contains info about a training job."""

    @staticmethod
    def new() -> "TrainingJobBuilder":
        return TrainingJobBuilder()

    # Stop function: accuracy[lookback_distance], average loss[lookback_distance], this_epoch,
    def __init__(
        self,
        group_name: str,
        dataset: DatasetInfo,
        stop_function: StopFunction,
        lookback_distance: int,
        batch_size: int,
        checkpoint_rate: int,
        learning_rate: float,
        testing_dataset: DatasetInfo | None,
        base_model_name: str | None,
    ):
        pass

    # Group to train
    # Starting model (or 'None' to create new)
    # Dataset to test on
    # Training hyperparameters
    # Stop condition(s)
    pass


class TrainingJobBuilder:
    """Constructs a training job."""

    def __init__(self):
        self._group_name: str | None = None
        self._dataset: DatasetInfo | None = None
        self._stop_function: StopFunction | None = None
        self._lookback_distance: int = 3
        self._batch_size: int = 1
        self._checkpoint_rate: int = 10
        self._learning_rate: float = 1e-4
        self._testing_dataset: DatasetInfo | None = None
        self._base_model_name: str | None = None

    def group_name(self, group_name: str) -> "TrainingJobBuilder":
        """Specifies the group name for the training job."""
        self._group_name = group_name
        return self

    def dataset(self, dataset: DatasetInfo) -> "TrainingJobBuilder":
        """Specifies the dataset for the training job."""
        self._dataset = dataset
        return self

    def stop_function(
        self, stop_function: Callable[[List[float], List[float], int], bool]
    ) -> "TrainingJobBuilder":
        self._stop_function = stop_function
        return self

    def lookback_distance(self, lookback_distance: int):
        self._lookback_distance = lookback_distance
        return self

    def batch_size(self, batch_size: int) -> "TrainingJobBuilder":
        self._batch_size = batch_size
        return self

    def checkpoint_rate(self, checkpoint_rate: int) -> "TrainingJobBuilder":
        self._checkpoint_rate = checkpoint_rate
        return self

    def learning_rate(self, learning_rate: float) -> "TrainingJobBuilder":
        self._learning_rate = learning_rate
        return self

    def testing_dataset(
        self, testing_dataset: DatasetInfo | None
    ) -> "TrainingJobBuilder":
        self._testing_dataset = testing_dataset
        return self

    def base_model_name(self, base_model_name: str | None) -> "TrainingJobBuilder":
        self._base_model_name = base_model_name
        return self

    def build(self) -> TrainingJob:
        if self._group_name is None:
            raise ValueError("A group name must be set.")
        if self._dataset is None:
            raise ValueError("A dataset must be set.")
        if self._stop_function is None:
            raise ValueError("A stop function must be set.")
        if self._lookback_distance < 0:
            raise ValueError("Lookback distance must be non-negative.")
        if self._batch_size < 1:
            raise ValueError("Batch size must be positive.")
        if self._checkpoint_rate < 1:
            raise ValueError("Checkpoint rate must be positive.")
        if self._learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        return TrainingJob(
            self._group_name,
            self._dataset,
            self._stop_function,
            self._lookback_distance,
            self._batch_size,
            self._checkpoint_rate,
            self._learning_rate,
            self._testing_dataset,
            self._base_model_name,
        )
