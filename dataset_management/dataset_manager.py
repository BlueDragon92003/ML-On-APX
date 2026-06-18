import pickle
from core.modes import Mode
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Set, Generic, TypeVar, Dict, Iterable, Type
from dataset_management.dataset_info import DatasetInfo
import torch


class Dataset(ABC, torch.utils.data.Dataset):
    @abstractmethod
    @classmethod
    def create(cls, components: Set[Tuple[Path, int]]) -> "Dataset":
        pass


class TreeNode:
    def __init__(self, name: str):
        self._name: str = name
        self._children: List["TreeNode"] = []

    def get_name(self) -> str:
        return self._name

    def get_children(self) -> List["TreeNode"]:
        return self._children

    def add_child(self, child: "TreeNode"):
        self._children.append(child)


T = TypeVar("T", bound=Dataset)


class DatasetManager(Generic[T]):
    """Manages a collection of datasets stored in a location."""

    def __init__(self, dataset_dir: Path, mode: Mode, dataset_class: Type[T]):
        self._root_dir_path = dataset_dir / mode.value / "root"
        self._sets_dir_path = dataset_dir / mode.value / "sets"
        self._sets_dir_path = dataset_dir / mode.value / "sets"
        self._dataset_class: Type[T] = dataset_class

    def __enter__(self):
        self._to_recompile: List[str] = []
        self._set_info: Dict[str, DatasetInfo]
        self._sources: TreeNode

        if not self._root_dir_path.exists():
            # Create
            self._root_dir_path.mkdir()
        elif not self._root_dir_path.is_dir():
            # Error
            raise NotADirectoryError(
                f"Expected a directory at `{self._root_dir_path}`."
            )

        if not self._sets_dir_path.exists():
            # Create
            self._sets_dir_path.mkdir()
        elif not self._sets_dir_path.is_dir():
            # Error
            raise NotADirectoryError(
                f"Expected a directory at `{self._sets_dir_path}`."
            )

        self._sources = fs_tree(
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

    def get_sources(self) -> TreeNode:
        return self._sources

    def create_dataset(self, name: str, dataset: DatasetInfo):
        """Create a new dataset."""
        if name in self._set_info.keys():
            raise ValueError("Set already exists!")
        self._set_info[name] = dataset
        self._to_recompile.append(name)

    def get_dataset_names(self) -> List[str]:
        """List all datasets."""
        return list(self._set_info.keys())

    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """Retrieve information for a dataset."""
        if dataset_name in self._set_info.keys():
            return self._set_info[dataset_name]
        raise ValueError(f"No such dataset {dataset_name}!")

    def get_dataset(self, dataset_name: str) -> T:
        """Retrieve a dataset object."""
        set_path = self._get_dataset_path(dataset_name)
        if dataset_name in self._to_recompile:
            self._recompile_dataset(set_path, self._set_info[dataset_name])
            self._to_recompile.remove(dataset_name)
        with open(set_path, mode="rb") as file:
            return pickle.load(file)

    def update_dataset(self, dataset_name: str, dataset_info: DatasetInfo):
        """Update a dataset's information."""
        self._to_recompile.append(dataset_name)
        self._set_info[dataset_name] = dataset_info

    def rename_dataset(self, dataset_name: str, new_name: str):
        """Rename a dataset."""
        path = self._get_dataset_path(dataset_name)
        path.move(self._get_dataset_path(new_name))
        info = self._set_info.pop(dataset_name)
        self._set_info[new_name] = info
        if dataset_name in self._to_recompile:
            self._to_recompile.remove(dataset_name)
            self._to_recompile.append(new_name)

    def delete_dataset(self, dataset_name: str):
        """Delete a dataset."""
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


def fs_tree(name: str, sources: Iterable[Path]) -> TreeNode:
    out = TreeNode(name)
    for source in sources:
        if not source.exists():
            continue
        elif source.is_dir():
            out.add_child(fs_tree(source.name, source.iterdir()))
        elif source.suffix == ".root":
            out.add_child(TreeNode(source.name))
    return out
