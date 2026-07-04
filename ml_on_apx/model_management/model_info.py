"""Information on specific models."""

from datetime import date, datetime
from typing import List

from ml_on_apx.labelling import Labels


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

        # TODO Properties for all of the above


class ModelInfo:
    """Stores training & testing info about a model."""

    def __init__(
        self,
        start_date: date,
        fork_time: datetime,
        labels: Labels,
        training_datasets: List[str],
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
        self._labels: Labels = labels
        self._training_datasets: List[str] = training_datasets
        # Testing information
        self._testing_information: List[ModelTestInfo] = []

        # TODO Properties for all of the above

    def add_testing_information(self, new_test: ModelTestInfo) -> None:
        """Add information about a test run on this model.

        Args:
            new_test (ModelTestInfo): The test that was run on this model.

        """
        self._testing_information.append(new_test)
