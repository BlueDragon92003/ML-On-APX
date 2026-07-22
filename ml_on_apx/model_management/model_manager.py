"""Manages models and model groups."""

import pickle
from pathlib import Path

import torch
from torch import nn

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
    PYTORCH_SUFFIX = ".pth"

    class ModelLookupError(LookupError):
        """A lookup error for a model."""

    class GroupLookupError(LookupError):
        """A lookup error for a model group."""

    def __init__(self, models_dir: Path, mode: Mode) -> None:
        """Create a new model manager for use in an environment.

        Args:
            models_dir (Path): The directory all model information is stored under.
            mode (Mode): If this manager is managing classification or identification
                models.

        """
        self._training_job: TrainingJob | None = None
        self._testing_job: TestingJob | None = None
        self._models_path: Path = models_dir / mode.value
        self._jobs_path: Path = self._models_path / self.JOBS_FILE

    @log_call(action_type="open" > _MANAGER)
    def __enter__(self) -> "ModelManager":
        """Initialize the environment.

        Returns:
            ModelManager: The fully-initialized model manager.

        """
        self._group_infos: dict[str, GroupInfo]
        self._model_infos: dict[str, dict[str, ModelInfo]]
        # Read Jobs
        if self._jobs_path.exists():
            with open(self._jobs_path, mode="rb") as file:
                self._training_job, self.testing_job = pickle.load(file)
        # For each folder:
        for file in self._models_path.iterdir():
            if not file.is_dir():
                continue  # not a group folder
            # Check for & read group_info.pckl
            group_name = file.name
            group_info_path = self.get_group_path(group_name) / self.GROUP_INFO_FILE
            models_info_path = self.get_group_path(group_name) / self.MODEL_INFOS_FILE
            if not (group_info_path.exists() and models_info_path.exists()):
                continue  # incomplete information in this folder
            with open(group_info_path, mode="rb") as file:
                group_info = pickle.load(file)
                self._group_infos.update({group_name: group_info})
            # Read model_infos.pckl
            with open(models_info_path, mode="rb") as file:
                models_info = pickle.load(file)
                self._model_infos.update({group_name: models_info})
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
        # Write Jobs
        with open(self._jobs_path, mode="wb") as file:
            pickle.dump((self._training_job, self._testing_job), file)
        # For each group info name:
        for group_name in self.group_names:
            group_info_path = self.get_group_path(group_name) / self.GROUP_INFO_FILE
            model_infos_path = self.get_group_path(group_name) / self.MODEL_INFOS_FILE
            # Pickle & write group_info.pckl
            with open(group_info_path, mode="wb") as file:
                pickle.dump(self._group_infos[group_name], file)
            # Pickle & write model_infos.pckl
            with open(model_infos_path, mode="wb") as file:
                pickle.dump(self._model_infos[group_name], file)
        return False

    @log_call(action_type="create" > _GROUP)
    def create_group(self, group_name: str, group_info: GroupInfo) -> None:
        """Create a new model group.

        Args:
            group_name (str): The name of the group to create.
            group_info (GroupInfo): The info of the group being create.

        """
        if group_name in self._model_infos[group_name].keys():
            raise ValueError(
                f"Model {group_name} already exists in group {group_name}!"
            )
        self._group_infos.update({group_name: group_info})
        self.get_group_path(group_name).mkdir()

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

        Raises:
            GroupLookupError: When the specified group name is not associated with a
                group.

        Returns:
            GroupInfo: The group info object.

        """
        if group_name not in self._group_infos.keys():
            raise self.GroupLookupError("No such group!")
        return self._group_infos[group_name]

    @log_call(action_type="rename" > _GROUP)
    def rename_group(self, group_name: str, new_name: str) -> None:
        """Change a group's name.

        Args:
            group_name (str): The name of the group to rename.
            new_name (str): The new name of the group.

        Raises:
            GroupLookupError: When the specified group name is not associated with a
                group.
            ValueError: When the new name provided is already in use.

        """
        if group_name not in self._group_infos.keys():
            raise self.GroupLookupError("No such group!")
        if new_name in self._group_infos.keys():
            raise ValueError(f"Name `{new_name}` already in use!")
        group_model_infos = self._model_infos.pop(group_name)
        self._model_infos.update({new_name: group_model_infos})

        group_info = self._group_infos.pop(group_name)
        self._group_infos.update({new_name: group_info})

        path = self.get_group_path(group_name)
        path.move(self.get_model_path(group_name, new_name))

    @log_call(action_type="name" > _GROUP)
    def delete_group(self, group_name: str) -> None:
        """Delete a group.

        Args:
            group_name (str): The group to delete.

        Raises:
            GroupLookupError: When the specified group name is not associated with a
                group.

        """
        if group_name not in self._group_infos.keys():
            raise self.GroupLookupError("No such group!")
        for model_name in self.get_model_names(group_name):
            self.delete_model(group_name, model_name)
        self._group_infos.pop(group_name)
        self._model_infos.pop(group_name)
        group_info_dir = self._models_path / group_name
        for file in group_info_dir.iterdir():
            file.unlink()
        group_info_dir.rmdir()

    @log_call(action_type="list" > _MODEL)
    def get_model_names(self, group_name: str) -> list[str]:
        """Return a list of tracked models.

        Args:
            group_name (str): The name of the group to get models from.

        Raises:
            GroupLookupError: When the specified group name is not associated with a
                group.

        Returns:
            List[str]: The list of models contained in this group.

        """
        if group_name not in self._group_infos.keys():
            raise self.GroupLookupError("No such group!")
        return list(self._model_infos[group_name].keys())

    @log_call(action_type="get" > _MODEL)
    def get_model_info(self, group_name: str, model_name: str) -> ModelInfo:
        """Return an existing model's settings with the provided name.

        Args:
            group_name (str): The name of the group to get a model from.
            model_name (str): The name of the model in that group to get.

        Raises:
            GroupLookupError: When the specified group name is not associated with a
                group.
            ModelLookupError: When the specified model name is not associated with a
                model in the group.

        Returns:
            ModelInfo: The info object for the model.

        """
        if group_name not in self._group_infos.keys():
            raise self.GroupLookupError("No such group!")
        if model_name not in self._model_infos[group_name].keys():
            raise self.ModelLookupError(f"No such model in group {group_name}!")
        return self._model_infos[group_name][model_name]

    def get_model(self, group_name: str, model_name: str) -> nn.Module:
        """Return an existing model's representation with the provided name.

        Args:
            group_name (str): The name of the group to get a model from.
            model_name (str): The name of the model in that group to get.

        Raises:
            GroupLookupError: When the specified group name is not associated with a
                group.
            ModelLookupError: When the specified model name is not associated with a
                model in the group.

        Returns:
            torch.nn.Module: The the model

        """
        if group_name not in self._group_infos.keys():
            raise self.GroupLookupError("No such group!")
        if model_name not in self._model_infos[group_name].keys():
            raise self.ModelLookupError(f"No such model in group {group_name}!")
        return torch.load(self.get_model_path(group_name, model_name))

    @log_call(action_type="create" > _MODEL)
    def create_model(
        self, group_name: str, model_name: str, model_info: ModelInfo, model: nn.Module
    ) -> None:
        """Create a new model.

        Args:
            group_name (str): The name of the group where the model is kept.
            model_name (str): The name of the model to rename.
            model_info (ModelInfo): The model's associated info.
            model (torch.nn.Module): The pytorch model.

        Raises:
            GroupLookupError: When the specified group name is not associated with a
                group.
            ValueError: When the specified model name already exists.

        """
        if group_name not in self._group_infos.keys():
            raise self.GroupLookupError("No such group!")
        if model_name in self._model_infos[group_name].keys():
            raise ValueError(
                f"Model {model_name} already exists in group {group_name}!"
            )
        self._model_infos[group_name].update({model_name: model_info})
        torch.save(model, self.get_model_path(group_name, model_name))

    @log_call(action_type="rename" > _MODEL)
    def rename_model(self, group_name: str, model_name: str, new_name: str) -> None:
        """Change a model's name.

        Args:
            group_name (str): The name of the group where the model is kept.
            model_name (str): The name of the model to rename.
            new_name (str): The new name of the model.

        Raises:
            GroupLookupError: When the specified group name is not associated with a
                group.
            ModelLookupError: When the specified model name is not associated with a
                model in the group.

        """
        if group_name not in self._group_infos.keys():
            raise self.GroupLookupError("No such group!")
        if model_name not in self._model_infos[group_name].keys():
            raise self.ModelLookupError(f"No such model in group {group_name}!")
        info = self._model_infos[group_name].pop(model_name)
        self._model_infos[group_name].update({new_name: info})
        path = self.get_model_path(group_name, model_name)
        path.move(self.get_model_path(group_name, new_name))

    @log_call(action_type="delete" > _MODEL)
    def delete_model(self, group_name: str, model_name: str) -> None:
        """Delete a model.

        Args:
            group_name (str): The name of the group the model to delete is saved.
            model_name (str): The name of the model to delete

        Raises:
            GroupLookupError: When the specified group name is not associated with a
                group.
            ModelLookupError: When the specified model name is not associated with a
                model in the group.

        """
        if group_name not in self._group_infos.keys():
            raise self.GroupLookupError("No such group!")
        if model_name not in self._model_infos[group_name].keys():
            raise self.ModelLookupError(f"No such model in group {group_name}!")
        self._model_infos[group_name].pop(model_name)
        path = self.get_model_path(group_name, model_name)
        if path.exists():
            path.unlink()

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

    @log_call(action_type="path" > _GROUP)
    def get_group_path(self, group_name: str) -> Path:
        """Get the path to a model specified by its name and group name.

        Args:
            group_name (str): The name of the model's group

        Returns:
            Path: The path to the model

        """
        return self._models_path / group_name

    @log_call(action_type="path" > _MODEL)
    def get_model_path(self, group_name: str, model_name: str) -> Path:
        """Get the path to a model specified by its name and group name.

        Args:
            group_name (str): The name of the model's group
            model_name (str): The name of the model.

        Returns:
            Path: The path to the model

        """
        return self.get_group_path(group_name) / (model_name + self.PYTORCH_SUFFIX)
