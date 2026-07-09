"""Manages models and model groups."""

from pathlib import Path

from ml_on_apx.logging import log_call
from ml_on_apx.model_management import _MODEL
from ml_on_apx.model_management.group_info import GroupInfo
from ml_on_apx.model_management.model_info import ModelInfo
from ml_on_apx.model_management.testing_job import TestingJob
from ml_on_apx.model_management.training_job import TrainingJob
from ml_on_apx.modes import Mode

_MANAGER = "manager" @ _MODEL
_GROUP = "group" @ _MANAGER
_MODEL = "model" @ _MANAGER

"""
models/
    classification/
        jobs.pckl
        ex_group/
            group_info.pckl
            model_infos.pckl
            ~checkpoint-2020-01-01-000.pth
            ⋮
            ⋮
            ex_saved_model_name.pth
            ex_2_saved_model.pth
            ⋮
            ⋮
        ⋮
        ⋮
"""


class ModelManager:
    """Manages a store of models saved to a file on disk."""

    PICKLE_SUFFIX = ".pckl"
    JOBS_FILE = "jobs" + PICKLE_SUFFIX
    GROUP_INFO_FILE = "group_info" + PICKLE_SUFFIX
    MODEL_INFOS_FILE = "model_infos" + PICKLE_SUFFIX

    def __init__(self, models_dir: Path, mode: Mode) -> None:
        """Create a new model manager for use in an environment.

        Args:
            models_dir (Path): The directory all model information is stored under.
            mode (Mode): If this manager is managing classification or identification
                models.

        """
        self._training_job = None
        self._testing_job = None
        self._models_path = models_dir / mode.value
        self._jobs_path = self._models_path / self.JOBS_FILE

    @log_call(action_type="open" > _MANAGER)
    def __enter__(self) -> "ModelManager":
        """Initialize the environment.

        Returns:
            ModelManager: The fully-initialized model manager.

        """
        # TODO
        self._group_infos: dict[str, GroupInfo]  # TODO
        self._model_infos: dict[str, dict[str, ModelInfo]]  # TODO
        self._dataset_names: set[str]
        return self

    @log_call(action_type="close" > _MANAGER)
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
        # TODO
        return False

    @log_call(action_type="create" > _GROUP)
    def create_group(self, group_name: str, group_info: GroupInfo) -> None:
        """Create a new model group.

        Args:
            group_name (str): The name of the group to create.
            group_info (GroupInfo): The info of the group being create.

        """

    @property
    def group_names(self) -> list[str]:
        """Get a list of group names tracked by this manager.

        Returns:
            List[str]: Group names tracked by this manager.

        """
        return list(self._group_infos.keys())

    @log_call(action_type="get" > _GROUP)
    def get_group_info(self, group_name: str) -> GroupInfo:
        """Get information about a specific group.

        Args:
            group_name (str): The name of the group.

        Returns:
            GroupInfo: The group info object.

        """  # TODO Raises
        if group_name not in self._group_infos.keys():
            raise LookupError("No such group!")
        return self._group_infos[group_name]

    @log_call(action_type="update" > _GROUP)
    def update_group(self, group_name: str, group_info: GroupInfo) -> None:
        """Modify the information on a specific group.

        Args:
            group_name (str): The name of the group to update
            group_info (GroupInfo): The object to update the group info to.

        """

    @log_call(action_type="rename" > _GROUP)
    def rename_group(self, group_name: str, new_name: str) -> None:
        """Change a group's name.

        Args:
            group_name (str): The name of the group to rename.
            new_name (str): The new name of the group.

        """

    @log_call(action_type="name" > _GROUP)
    def delete_group(self, group_name: str) -> None:
        """Delete a group.

        Args:
            group_name (str): The group to delete.

        """

    @log_call(action_type="list" > _MODEL)
    def get_model_names(self, group_name: str) -> list[str]:
        """Return a list of tracked models.

        Args:
            group_name (str): The name of the group to get models from.

        Returns:
            List[str]: The list of models contained in this group.

        """

    @log_call(action_type="get" > _MODEL)
    def get_model(self, group_name: str, model_name: str) -> ModelInfo:
        """Return an existing model's settings with the provided name.

        Args:
            group_name (str): The name of the group to get a model from.
            model_name (str): The name of the model in that group to get.

        Returns:
            ModelInfo: The info object for the model.

        """  # TODO raises
        if group_name not in self._group_infos.keys():
            raise LookupError()  # TODO
        if model_name not in self._model_infos[group_name].keys():
            raise LookupError()  # TODO
        return self._model_infos[group_name][model_name]

    @log_call(action_type="rename" > _MODEL)
    def rename_model(self, group_name: str, model_name: str, new_name: str) -> None:
        """Change a model's name.

        Args:
            group_name (str): The name of the group where the model is kept.
            model_name (str): The name of the model to rename.
            new_name (str): The new name of the model.

        """

    @log_call(action_type="delete" > _MODEL)
    def delete_model(self, group_name: str, model_name: str) -> None:
        """Delete a model.

        Args:
            group_name (str): The name of the group the model to delete is saved.
            model_name (str): The name of the model to delete

        """

    @property
    def training_job(self) -> TrainingJob | None:
        """The current training job."""
        return self._training_job

    @training_job.setter
    def training_job(self, job: TrainingJob | None) -> None:
        self._training_job = job

    @property
    def testing_job(self) -> TestingJob | None:
        """The current testing job."""
        return self._testing_job

    @testing_job.setter
    def testing_job(self, job: TestingJob | None) -> None:
        self._testing_job = job
