"""Manages the dataset pickles and which datasets are available."""

import pickle
from pathlib import Path
from typing import ClassVar, Dict, Generic, List, Set, Type, TypeVar

from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.dataset_management.dataset_info import DatasetInfo
from ml_on_apx.dataset_management.tree import TreeNode
from ml_on_apx.logging import log_call
from ml_on_apx.modes import Mode

ManagedDataset = TypeVar("ManagedDataset", bound=Dataset)


class DatasetManager(Generic[ManagedDataset]):
    """Manages a collection of datasets stored in a location."""

    ROOT_DIR_NAME = "root"
    SETS_DIR_NAME = "sets"
    PICKLE_SUFFIX = ".pckl"
    SET_INFO_NAME = "_set_info"
    DATASET_PICKLE_SUFFIXES: ClassVar[list[str]] = [".dataset", PICKLE_SUFFIX]

    def __init__(
        self, dataset_dir: Path, mode: Mode, dataset_class: Type[ManagedDataset]
    ) -> None:
        """Create a new dataset manager for use in an environment.

        Args:
            dataset_dir (Path): The directory all dataset information is stored under.
            mode (Mode): The mode of datasets the manager is managing.
            dataset_class (Type[T]): The class that should be used for datasets.

        """
        self._root_dir_path = dataset_dir / mode.value / DatasetManager.ROOT_DIR_NAME
        self._sets_dir_path = dataset_dir / mode.value / DatasetManager.SETS_DIR_NAME
        self._set_info_path = dataset_dir / (
            str(mode.value)
            + DatasetManager.SET_INFO_NAME
            + DatasetManager.PICKLE_SUFFIX
        )
        self._dataset_class: Type[ManagedDataset] = dataset_class

    @log_call(action_type="data:manager:open")
    def __enter__(self) -> "DatasetManager[ManagedDataset]":
        """Initialize the environment.

        Raises:
            NotADirectoryError: If files exist with the names of directories the manager
                expects.

        Returns:
            DatasetManager[T]: The fully-initialized dataset manager.

        """
        self._to_recompile: List[str] = []
        self._set_info: Dict[str, DatasetInfo]
        self._sources: TreeNode

        self._set_info: Dict[str, DatasetInfo] = {}

        if self._set_info_path.exists() and self._set_info_path.is_file():
            with open(self._set_info_path, mode="rb") as set_info:
                self._set_info: Dict[str, DatasetInfo] = pickle.load(set_info)
        elif self._set_info_path.exists():
            # TODO LOG WARN
            pass

        if not self._root_dir_path.exists():
            # Create
            self._root_dir_path.mkdir(parents=True)
        elif not self._root_dir_path.is_dir():
            # Error
            # TODO LOG CRITICAL
            raise NotADirectoryError(
                f"Expected a directory at `{self._root_dir_path}`."
            )

        if not self._sets_dir_path.exists():
            # Create
            self._sets_dir_path.mkdir(parents=True)
        elif not self._sets_dir_path.is_dir():
            # Error
            # TODO LOG CRITICAL
            raise NotADirectoryError(
                f"Expected a directory at `{self._sets_dir_path}`."
            )

        self._sources = TreeNode.from_filesystem(
            self._root_dir_path.parent.name, self._root_dir_path.iterdir()
        )

        valid = self._datasets_in_dir(self._sets_dir_path)
        for set in list(self._set_info.keys()):
            if set not in valid:
                self._set_info.pop(set)

        return self

    @log_call(action_type="data:manager:close")
    def __exit__(self, _exc_type, _exc_value, _exc_traceback) -> bool:  # noqa: ANN001
        """Safely leave a managed environment.

        Args:
            _exc_type: The type of the exception that occured in this
                environment, if any.
            _exc_value: The exception that occured in this environment, if any.
            _exc_traceback: The traceback of the exception.

        Propogates:
            Any errors from dataset creation.

        Returns:
            bool: If the exception was handled (always False)

        """
        count_to_recompile = len(self._to_recompile)
        done = 0
        if count_to_recompile > 0:
            print("(Re)compiling datasets. Please wait.")
            to_finish = 77.0 / count_to_recompile
            print(f"[={' ' * 77}]", end="\r")
        for dataset_name in self._to_recompile:
            self._recompile_dataset(
                self._get_dataset_path(dataset_name), self._set_info[dataset_name]
            )
            done += 1
            eqls = int(done * to_finish)
            print(f"[={'=' * eqls}{' ' * (77 - eqls)}]", end="\r")
        if done:
            print()
        if not self._set_info_path.exists() or (
            self._set_info_path.exists() and self._set_info_path.is_file()
        ):
            with open(self._set_info_path, mode="wb") as set_info:
                pickle.dump(self._set_info, set_info)
        else:
            # TODO LOG ERROR
            pass
        return False

    @log_call(action_type="data:manager:get_root_path")
    def get_root_dir_path(self) -> Path:
        """Get the path to the ROOT file directory."""
        return self._root_dir_path

    @log_call(action_type="data:manager:get_sources")
    def get_sources(self) -> TreeNode:
        """Get the possible ROOT sources as a tree."""
        return self._sources

    @log_call(action_type="data:manager:create")
    def create_dataset(self, name: str, dataset: DatasetInfo) -> None:
        """Create a new dataset.

        Args:
            name (str): The name of the dataset to create.
            dataset (DatasetInfo): The information for the dataset.

        Raises:
            ValueError: If a dataset with that name already exists.

        """
        if name in self._set_info.keys():
            raise ValueError("Set already exists!")
        self._set_info[name] = dataset
        self._to_recompile.append(name)

    @log_call(action_type="data:manager:list_names")
    def get_dataset_names(self) -> Set[str]:
        """List all datasets this manager is aware of."""
        return set(self._set_info.keys())

    @log_call(action_type="data:manager:get_info")
    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """Retrieve information for a dataset.

        Args:
            dataset_name (str): The name of the dataset data is to be retrieved for.

        Raises:
            LookupError: If the provided dataset name is not tied to a dataset.

        Returns:
            DatasetInfo: The information for the dataset.

        """
        if dataset_name not in self._set_info.keys():
            raise LookupError("No such set!")
        return self._set_info[dataset_name]

    @log_call(action_type="data:manager:get")
    def get_dataset(self, dataset_name: str) -> ManagedDataset:
        """Retrieve a dataset object.

        Args:
            dataset_name (str): The name of the dataset to be retrieved.

        Raises:
            LookupError: If the provided dataset name is not associated with a dataset.

        Returns:
            ManagedDataset: The dataset object.

        """
        if dataset_name not in self._set_info.keys():
            raise LookupError("No such set!")
        set_path = self._get_dataset_path(dataset_name)
        if dataset_name in self._to_recompile:
            self._recompile_dataset(set_path, self._set_info[dataset_name])
            self._to_recompile.remove(dataset_name)
        with open(set_path, mode="rb") as file:
            return pickle.load(file)

    @log_call(action_type="data:manager:update")
    def update_dataset(self, dataset_name: str, dataset_info: DatasetInfo) -> None:
        """Update the information for the provided dataset's name.

        Args:
            dataset_name (str): The name of the dataset to update.
            dataset_info (DatasetInfo): The information it is being updated to.

        Raises:
            LookupError: If the provided dataset name is not associated with a dataset.

        """
        if dataset_name not in self._set_info.keys():
            raise LookupError("No such set!")
        self._to_recompile.append(dataset_name)
        self._set_info[dataset_name] = dataset_info

    @log_call(action_type="data:manager:rename")
    def rename_dataset(self, dataset_name: str, new_name: str) -> None:
        """Rename a dataset.

        Args:
            dataset_name (str): The name of the dataset to update.
            new_name (str): The new name for the dataset.

        Raises:
            LookupError: If the provided dataset name is not associated with a dataset.
            ValueError: If the new name matches an existing dataset.

        """
        if dataset_name not in self._set_info.keys():
            raise LookupError(f"No such set `{dataset_name}`!")
        if new_name in self._set_info.keys():
            raise ValueError(f"Set `{new_name}` already exists!")
        path = self._get_dataset_path(dataset_name)
        if path.exists():
            path.move(self._get_dataset_path(new_name))
        info = self._set_info.pop(dataset_name)
        self._set_info[new_name] = info
        if dataset_name in self._to_recompile:
            self._to_recompile.remove(dataset_name)
            self._to_recompile.append(new_name)

    @log_call(action_type="data:manager:delete")
    def delete_dataset(self, dataset_name: str) -> None:
        """Delete a dataset.

        Args:
            dataset_name (str): The dataset to delete.

        Raises:
            LookupError: If the dataset does not exist.

        """
        if dataset_name not in self._set_info.keys():
            raise LookupError(f"No such set `{dataset_name}`!")
        path = self._get_dataset_path(dataset_name)
        if path.exists():
            path.unlink()
        self._set_info.pop(dataset_name)
        if dataset_name in self._to_recompile:
            self._to_recompile.remove(dataset_name)

    @log_call(action_type="data:manager:recompile")
    def _recompile_dataset(self, path: Path, dataset_info: DatasetInfo) -> None:
        """Create and pickle a dataset based on the provided information.

        Propogates:
            Any errors from dataset creation.

        Args:
            path (Path): The path to save the dataset to.
            dataset_info (DatasetInfo): The information to compile the dataset with.

        """
        to_pickle = self._dataset_class.create(dataset_info.get_numbered_sources())
        with open(path, mode="wb") as file:
            pickle.dump(to_pickle, file)

    def _get_dataset_path(self, dataset_name: str) -> Path:
        """Get the path to a dataset.

        Args:
            dataset_name (str): The name of the hypothetical dataset.

        Returns:
            Path: The path where that dataset would be stored.

        """
        return (
            self._sets_dir_path
            / f"{dataset_name}{''.join(DatasetManager.DATASET_PICKLE_SUFFIXES)}"
        )

    def _datasets_in_dir(self, dir: Path) -> List[str]:
        """Return a list of datasets pickled to files.

        Args:
            dir (Path): The directory to look under.

        Returns:
            List[str]: A list of dataset names in the path.

        """

        def _filter(path: Path) -> bool:
            return (
                path.is_file()
                and path.suffixes == DatasetManager.DATASET_PICKLE_SUFFIXES
            )

        options = dir.iterdir()
        correct_type_and_extention = filter(
            _filter,
            options,
        )
        just_the_name = (
            path.name.removesuffix("".join(DatasetManager.DATASET_PICKLE_SUFFIXES))
            for path in correct_type_and_extention
        )
        return list(just_the_name)
