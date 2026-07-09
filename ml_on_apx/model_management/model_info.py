"""Information on specific models."""

from datetime import date, datetime

from ml_on_apx.logging import log_call
from ml_on_apx.model_management import _MODEL

_MODEL_INFO = "model" @ _MODEL


class ModelTestInfo:
    """Stores data corresponding to a single model test."""

    def __init__(
        self,
        test_time: datetime,
        accuracy: float,
        average_loss: float,
        run_by_user: bool = False,
    ) -> None:
        """Create a new object to store information on a model's test.

        Args:
            test_time (datetime): When the test started.
            accuracy (float): The accuracy of the model.
            average_loss (float): The average loss of the model accross the test.
            run_by_user (bool, optional): If the test was run by the user or during the
                training process. Defaults to False.

        """
        self._test_time: datetime = test_time
        self._run_by_user: bool = run_by_user
        self._accuracy: float = accuracy
        self._average_loss: float = average_loss

        @property
        def test_time(self: "ModelTestInfo") -> datetime:
            """When the test was run."""
            return self._test_time

        @property
        def run_by_user(self: "ModelTestInfo") -> bool:
            """If the test was run by the user or as part of the checkpoint process."""
            return self._run_by_user

        @property
        def accuracy(self: "ModelTestInfo") -> float:
            """The test's accuracy."""
            return self._accuracy

        @property
        def average_loss(self: "ModelTestInfo") -> float:
            """The average loss across the test."""
            return self._average_loss


class ModelInfo:
    """Stores training & testing info about a model."""

    def __init__(
        self,
        start_date: date,
        fork_time: datetime,
        labels: str,
        training_datasets: list[str],
    ) -> None:
        """Create a new object to store information on a model.

        Args:
            start_date (date): When the model training cycle started.
            fork_time (datetime): When this set of weights was extracted from the
                training process.
            labels (Labels): The labels this model uses.
            training_datasets (List[str]): The dataset namess the model pulled from.

        """
        # Training information
        self._training_start_date: date = start_date
        self._model_fork_time: datetime = fork_time
        self._group: str = labels
        self._training_datasets: list[str] = training_datasets
        # Testing information
        self._testing_information: list[ModelTestInfo] = []

    @property
    def training_start_date(self: "ModelInfo") -> date:
        """The date the training job started."""
        return self._training_start_date

    @property
    def model_fork_time(self: "ModelInfo") -> datetime:
        """When this model was saved."""
        return self._model_fork_time

    @property
    def group(self: "ModelInfo") -> str:
        """The name of group this model belongs to."""
        return self._group

    @property
    def training_datasets(self: "ModelInfo") -> list[str]:
        """The date the training job started."""
        return self._training_datasets

    @property
    def testing_information(self: "ModelInfo") -> list[ModelTestInfo]:
        """The date the training job started."""
        return self._testing_information

    @log_call(action_type="add" > _MODEL_INFO)
    def add_testing_information(self, new_test: ModelTestInfo) -> None:
        """Add information about a test run on this model.

        Args:
            new_test (ModelTestInfo): The test that was run on this model.

        """
        self._testing_information.append(new_test)
