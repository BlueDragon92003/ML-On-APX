from dataset_management.dataset_info import DatasetInfo
from pathlib import Path


class TestingJob:
    """Contains info about a testing job."""

    def __init__(self, target: Path, dataset: DatasetInfo):
        self.target: Path = target
        self.dataset: DatasetInfo = dataset
