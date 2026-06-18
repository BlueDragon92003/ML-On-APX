import pickle
from ml_on_apx.modes import Mode
from pathlib import Path
from typing import List, Generic, TypeVar, Dict, Type
from ml_on_apx.dataset_management.dataset_info import DatasetInfo
from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.dataset_management.tree import TreeNode

T = TypeVar("T", bound=Dataset)


class DatasetManager(Generic[T]):
    """Manages a collection of datasets stored in a location."""

    def __init__(self, dataset_dir: Path, mode: Mode, dataset_class: Type[T]):
        self._root_dir_path = dataset_dir / mode.value / "root"
        self._sets_dir_path = dataset_dir / mode.value / "sets"
        self._set_info_path = dataset_dir / f"{mode.value}_set_info.pckl"
        self._dataset_class: Type[T] = dataset_class

    def __enter__(self):
        self._to_recompile: List[str] = []
        self._set_info: Dict[str, DatasetInfo]
        self._sources: TreeNode

        self._set_info: Dict[str, DatasetInfo] = dict()
        if self._set_info_path.exists() and self._set_info_path.is_file():
            with open(self._set_info_path, mode="rb") as set_info:
                self._set_info: Dict[str, DatasetInfo] = pickle.load(set_info)
        elif self._set_info_path.exists():
            # TODO LOG WARN
            pass

        if not self._root_dir_path.exists():
            # Create
            self._root_dir_path.mkdir()
        elif not self._root_dir_path.is_dir():
            # Error
            # TODO LOG CRITICAL
            raise NotADirectoryError(
                f"Expected a directory at `{self._root_dir_path}`."
            )

        if not self._sets_dir_path.exists():
            # Create
            self._sets_dir_path.mkdir()
        elif not self._sets_dir_path.is_dir():
            # Error
            # TODO LOG CRITICAL
            raise NotADirectoryError(
                f"Expected a directory at `{self._sets_dir_path}`."
            )

        self._sources = TreeNode.from_filesystem(
            self._root_dir_path.parent.name, self._root_dir_path.iterdir()
        )

        valid = list(self._sets_dir_path.iterdir())
        for set in self._set_info.keys():
            if set not in valid:
                self._set_info.pop(set)

    def __exit__(self):
        for dataset_name in self._to_recompile:
            self._recompile_dataset(
                self._get_dataset_path(dataset_name), self._set_info[dataset_name]
            )
        if self._set_info_path.exists() and self._set_info_path.is_file():
            with open(self._set_info_path, mode="wb") as set_info:
                pickle.dump(self._set_info, set_info)
        else:
            # TODO LOG ERROR
            pass

    def get_sources(self) -> TreeNode:
        return self._sources

    def create_dataset(self, name: str, dataset: DatasetInfo):
        """Create a new dataset."""
        if name in self._set_info.keys():
            # TODO LOG WARN
            raise KeyError("Set already exists!")
        self._set_info[name] = dataset
        self._to_recompile.append(name)

    def get_dataset_names(self) -> List[str]:
        """List all datasets."""
        return list(self._set_info.keys())

    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """Retrieve information for a dataset."""
        return self._set_info[dataset_name]

    def get_dataset(self, dataset_name: str) -> T:
        """Retrieve a dataset object."""
        if dataset_name not in self._set_info.keys():
            raise KeyError("No such set!")
        set_path = self._get_dataset_path(dataset_name)
        if dataset_name in self._to_recompile:
            self._recompile_dataset(set_path, self._set_info[dataset_name])
            self._to_recompile.remove(dataset_name)
        with open(set_path, mode="rb") as file:
            return pickle.load(file)

    def update_dataset(self, dataset_name: str, dataset_info: DatasetInfo):
        """Update a dataset's information."""
        if dataset_name not in self._set_info.keys():
            raise KeyError("No such set!")
        self._to_recompile.append(dataset_name)
        self._set_info[dataset_name] = dataset_info

    def rename_dataset(self, dataset_name: str, new_name: str):
        """Rename a dataset."""
        if dataset_name not in self._set_info.keys():
            raise KeyError(f"No such set `{dataset_name}`!")
        if new_name in self._set_info.keys():
            raise KeyError(f"Set `{new_name}` already exists!")
        path = self._get_dataset_path(dataset_name)
        path.move(self._get_dataset_path(new_name))
        info = self._set_info.pop(dataset_name)
        self._set_info[new_name] = info
        if dataset_name in self._to_recompile:
            self._to_recompile.remove(dataset_name)
            self._to_recompile.append(new_name)

    def delete_dataset(self, dataset_name: str):
        """Delete a dataset."""
        if dataset_name not in self._set_info.keys():
            raise KeyError(f"No such set `{dataset_name}`!")
        path = self._get_dataset_path(dataset_name)
        path.unlink()
        self._set_info.pop(dataset_name)
        if dataset_name in self._to_recompile:
            self._to_recompile.remove(dataset_name)

    def _recompile_dataset(self, path: Path, dataset_info: DatasetInfo):
        """(Re)create a dataset, pickle it, and save it to the path."""
        to_pickle = self._dataset_class.create(dataset_info.get_labeled_sources())
        with open(path, mode="wb") as file:
            pickle.dump(to_pickle, file)

    def _get_dataset_path(self, dataset_name: str) -> Path:
        return self._sets_dir_path / f"{dataset_name}.root"
