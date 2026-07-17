"""A job to train a model."""

from ml_on_apx.logging import log_call
from ml_on_apx.model_management import _MODEL
from ml_on_apx.model_management.stop_functions import StopFunction

_TRAINING = "training" @ _MODEL


class TrainingJob:
    """Contains info about a training job."""

    @staticmethod
    def new() -> "TrainingJobBuilder":
        """Create a new training job builder."""
        return TrainingJobBuilder()

    # Stop function: accuracy[lookback_distance], average loss[lookback_distance],
    # this_epoch,
    def __init__(
        self,
        group_name: str,
        dataset: str,
        stop_function: StopFunction,
        lookback_distance: int,
        batch_size: int,
        checkpoint_rate: int,
        learning_rate: float,
        testing_dataset: str | None,
        base_model_name: str | None,
    ) -> None:
        """Initialize a new training job.

        This method should not be directly called. Use `new` to build a job.

        Args:
            group_name (str): The model group to train around.
            dataset (str): The name of the dataset to train against.
            stop_function (StopFunction): The function the job uses to determine whether
                to stop.
            lookback_distance (int): The length of history to provide to the stop
                function.
            batch_size (int): Hyperparameter: The batch size to train with.
            checkpoint_rate (int): Hyperparameter: How often to make a checkpoint.
            learning_rate (float): Hyperparameter: The learning rate for the model.
            testing_dataset (str | None): The name of the dataset to test against, or
                None if the same dataset used for training is to be used.
            base_model_name (str | None): A model in the group to pull staring weights
                from.

        """
        self._group_name = group_name
        self._dataset = dataset
        self._stop_function = stop_function
        self._lookback_distance = lookback_distance
        self._batch_size = batch_size
        self._checkpoint_rate = checkpoint_rate
        self._learning_rate = learning_rate
        self._testing_dataset = testing_dataset
        self._base_model_name = base_model_name

    @property
    def group_name(self: "TrainingJob") -> str:
        """The name of the model group to train."""
        return self._group_name

    @property
    def dataset(self: "TrainingJob") -> str:
        """The name of the dataset to train against."""
        return self._dataset

    @property
    def stop_function(self: "TrainingJob") -> StopFunction:
        """The function used to determine if the training job should stop."""
        return self._stop_function

    @property
    def lookback_distance(self: "TrainingJob") -> int:
        """How far back each of the parameters to the stop function should go."""
        return self._lookback_distance

    @property
    def batch_size(self: "TrainingJob") -> int:
        """Hyperparemeter. The size of the training batch."""
        return self._batch_size

    @property
    def checkpoint_rate(self: "TrainingJob") -> int:
        """How many epochs between testing & checkpointing."""
        return self._checkpoint_rate

    @property
    def learning_rate(self: "TrainingJob") -> float:
        """Hyperparameter. The learning rate of the model."""
        return self._learning_rate

    @property
    def testing_dataset(self: "TrainingJob") -> str | None:
        """The name of the dataset to test with, if different than the training set."""
        return self._testing_dataset

    @property
    def base_model_name(self: "TrainingJob") -> str | None:
        """The name of the model in this group to copy weights from as a template."""
        return self._base_model_name


class TrainingJobBuilder:
    """Constructs a training job."""

    def __init__(self) -> None:
        """Initialize the builder."""
        self._group_name: str | None = None
        self._dataset: str | None = None
        self._stop_function: StopFunction | None = None
        self._lookback_distance: int = 3
        self._batch_size: int = 1
        self._checkpoint_rate: int = 10
        self._learning_rate: float = 1e-4
        self._testing_dataset: str | None = None
        self._base_model_name: str | None = None

    def group_name(self, group_name: str) -> None:
        """Specify the group name for the training job.

        Args:
            group_name (str): The name of the group.

        """
        self._group_name = group_name

    def dataset(self, dataset: str) -> None:
        """Specify the dataset for the training job.

        Args:
            dataset (str): The name of the dataset to use.

        """
        self._dataset = dataset

    def stop_function(self, stop_function: StopFunction) -> None:
        """Specify the stop function for the training job.

        Args:
            stop_function (StopFunction): The stop function.

        """
        self._stop_function = stop_function

    def lookback_distance(self, lookback_distance: int) -> None:
        """Specify the lookback distance for the training job.

        Args:
            lookback_distance (int): The lookback distance.

        """
        self._lookback_distance = lookback_distance

    def batch_size(self, batch_size: int) -> None:
        """Specify the batch size for the training job.

        Args:
            batch_size (int): The batch size.

        """
        self._batch_size = batch_size

    def checkpoint_rate(self, checkpoint_rate: int) -> None:
        """Specify the checkpoint rate for the training job.

        Args:
            checkpoint_rate (int): The checkpoint rate.

        """
        self._checkpoint_rate = checkpoint_rate

    def learning_rate(self, learning_rate: float) -> None:
        """Specify the learning rate for the training job.

        Args:
            learning_rate (float): The learning rate.

        """
        self._learning_rate = learning_rate

    def testing_dataset(self, testing_dataset: str | None) -> None:
        """Specify the testing dataset for the training job.

        Args:
            testing_dataset (str | None): The name of the testing dataset.

        """
        self._testing_dataset = testing_dataset

    def base_model_name(self, base_model_name: str | None) -> None:
        """Specify the base model for the training job.

        Args:
            base_model_name (str | None): The name of the base model.

        """
        self._base_model_name = base_model_name

    @log_call(action_type="build_job" > _TRAINING)
    def build(self) -> TrainingJob:
        """Build a training job from this builder.

        Raises:
            TypeError: If a required component was not provided.
            ValueError: If an improper value for a component was provided.

        Returns:
            TrainingJob: The built training job.

        """
        if self._group_name is None:
            raise TypeError("A group name must be set.")
        if self._dataset is None:
            raise TypeError("A dataset must be set.")
        if self._stop_function is None:
            raise TypeError("A stop function must be set.")
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
