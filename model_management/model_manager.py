from model_management.group_info import GroupInfo
from pathlib import Path
from model_management.testing_job import TestingJob
from model_management.training_job import TrainingJob
from typing import List
from model_management.model_info import ModelInfo


class ModelManager:
    """Manages a store of models saved to a file on disk."""

    def __init__(self, models_dir: Path):
        pass

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def create_group(self, group_name: str, group_info: GroupInfo):
        """Create a new model group."""
        pass

    def get_group_names(self) -> List[str]:
        """Get a list of all groups."""
        pass

    def get_group_info(self, group_name: str) -> GroupInfo:
        """Get information about a specific group."""
        pass

    def update_group(self, group_name: str, group_info: GroupInfo):
        """Modify the information on a specific group."""
        pass

    def rename_group(self, group_name: str, new_name: str):
        """Change a group's name."""
        pass

    def delete_group(self, group_name: str):
        """Delete a group."""
        pass

    def get_model_names(self, group_name: str) -> List[str]:
        """Return a list of tracked models."""
        pass

    def get_model(self, group_name: str, model_name: str) -> ModelInfo:
        """Return an existing model's settings with the provided name."""
        pass

    def rename_model(self, group_name: str, model_name: str, new_name: str):
        """Change a model's name."""
        pass

    def delete_model(self, group_name: str, model_name: str) -> bool:
        """Delete a model."""
        pass

    def get_training_job(self) -> TrainingJob:
        """Get the current training job."""
        pass

    def set_training_job(self, job: TrainingJob):
        """Set the current training job."""
        pass

    def get_testing_job(self) -> TestingJob:
        """get the current testing job."""
        pass

    def set_testing_job(self, job: TestingJob):
        pass
