"""Manages models and model groups."""

from pathlib import Path

from ml_on_apx.model_management.group_info import GroupInfo
from ml_on_apx.model_management.model_info import ModelInfo
from ml_on_apx.model_management.testing_job import TestingJob
from ml_on_apx.model_management.training_job import TrainingJob


class ModelManager:
    """Manages a store of models saved to a file on disk."""

    def __init__(self, models_dir: Path) -> None:
        """Create a new model manager for use in an environment.

        Args:
            models_dir (Path): The directory all model information is stored under.

        """

    def __enter__(self) -> "ModelManager":
        """Initialize the environment.

        Returns:
            ModelManager: The fully-initialized model manager.

        """

    def __exit__(self, _exc_type, _exc_value, _exc_traceback) -> bool:  # noqa: ANN001
        """Safely leave a managed environment.

        Args:
            _exc_type: The type of the exception that occured in this
                environment, if any.
            _exc_value: The exception that occured in this environment, if any.
            _exc_traceback: The traceback of the exception.

        Returns:
            bool: If the exception was handled (always False)

        """

    def create_group(self, group_name: str, group_info: GroupInfo) -> None:
        """Create a new model group.

        Args:
            group_name (str): The name of the group to create.
            group_info (GroupInfo): The info of the group being create.

        """

    def get_group_names(self) -> list[str]:
        """Get a list of group names tracked by this manager.

        Returns:
            List[str]: Group names tracked by this manager.

        """

    def get_group_info(self, group_name: str) -> GroupInfo:
        """Get information about a specific group.

        Args:
            group_name (str): The name of the group.

        Returns:
            GroupInfo: The group info object.

        """

    def update_group(self, group_name: str, group_info: GroupInfo) -> None:
        """Modify the information on a specific group.

        Args:
            group_name (str): The name of the group to update
            group_info (GroupInfo): The object to update the group info to.

        """

    def rename_group(self, group_name: str, new_name: str) -> None:
        """Change a group's name.

        Args:
            group_name (str): The name of the group to rename.
            new_name (str): The new name of the group.

        """

    def delete_group(self, group_name: str) -> None:
        """Delete a group.

        Args:
            group_name (str): The group to delete.

        """

    def get_model_names(self, group_name: str) -> list[str]:
        """Return a list of tracked models.

        Args:
            group_name (str): The name of the group to get models from.

        Returns:
            List[str]: The list of models contained in this group.

        """

    def get_model(self, group_name: str, model_name: str) -> ModelInfo:
        """Return an existing model's settings with the provided name.

        Args:
            group_name (str): The name of the group to get a model from.
            model_name (str): The name of the model in that group to get.

        Returns:
            ModelInfo: The info object for the model.

        """

    def rename_model(self, group_name: str, model_name: str, new_name: str) -> None:
        """Change a model's name.

        Args:
            group_name (str): The name of the group where the model is kept.
            model_name (str): The name of the model to rename.
            new_name (str): The new name of the model.

        """

    def delete_model(self, group_name: str, model_name: str) -> None:
        """Delete a model.

        Args:
            group_name (str): The name of the group the model to delete is saved.
            model_name (str): The name of the model to delete

        """

    def get_training_job(self) -> TrainingJob:
        """Get the current training job.

        Returns:
            TrainingJob: The current training job.

        """

    def set_training_job(self, job: TrainingJob) -> None:
        """Set the training job.

        Args:
            job (TrainingJob): The new training job.

        """

    def get_testing_job(self) -> TestingJob:
        """Get the current testing job.

        Returns:
            TestingJob: The current testing job.

        """

    def set_testing_job(self, job: TestingJob) -> None:
        """Set the testing job.

        Args:
            job (TestingJob): The new testing job.

        """
