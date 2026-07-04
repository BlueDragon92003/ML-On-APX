"""A job to test a model."""

from pathlib import Path

from ml_on_apx.dataset_management.dataset_info import DatasetInfo


class TestingJob:
    """Contains info about a testing job."""

    def __init__(self, target: Path, dataset: DatasetInfo) -> None:
        """Create a new testing job.

        Args:
            target (Path): The path to the model to test.
            dataset (DatasetInfo): The dataset to test against.

        """
        self.target: Path = target
        self.dataset: DatasetInfo = dataset
