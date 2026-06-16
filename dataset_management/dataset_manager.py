from pathlib import Path
from typing import List
from dataset_management.dataset_info import DatasetInfo


class DatasetManager:
    """Manages a collection of datasets stored in a location."""

    def __init__(self, dataset_dir: Path):
        pass

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def create_dataset(self, dataset: DatasetInfo):
        """Create a new dataset."""
        pass

    def get_dataset_names(self) -> List[str]:
        """List all datasets."""
        pass

    def get_dataset(self, dataset_name: str) -> DatasetInfo:
        """Retrieve information for a dataset."""
        pass

    def update_dataset(self, dataset_name: str, dataset: DatasetInfo):
        """Update a dataset's information."""
        pass

    def rename_dataset(self, dataset_name: str, new_name: str):
        """Rename a dataset."""
        pass

    def delete_dataset(self, dataset_name: str):
        """Delete a dataset."""
        pass
