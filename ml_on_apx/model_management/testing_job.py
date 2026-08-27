"""A job to test a model."""


class TestingJob:
    """Contains info about a testing job."""

    def __init__(self, target: tuple[str, str], dataset: str) -> None:
        """Create a new testing job.

        Args:
            target (Path): The path to the model to test.
            dataset (str): The dataset to test against.

        """
        self._target: tuple[str, str] = target
        self._dataset: str = dataset

    @property
    def target(self: "TestingJob") -> tuple[str, str]:
        """The path to the model to test."""
        return self._target

    @property
    def dataset(self: "TestingJob") -> str:
        """The name of the dataset to test using."""
        return self._dataset
