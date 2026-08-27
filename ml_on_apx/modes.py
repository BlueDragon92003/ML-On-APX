"""Modes that this application runs in."""

from enum import Enum
from pathlib import Path
from typing import Type

from ml_on_apx.cluster_classification.cluster_classification_dataset import (
    ClusterClassificationDataset,
)
from ml_on_apx.dataset_management.dataset import Dataset


class _MockDataset(Dataset):
    def __init__(self, components: set[tuple[Path, int]]) -> None:
        self._components = components

    def __eq__(self, other: object) -> bool:
        if type(other) is not _MockDataset:
            return False
        return self._components == other._components

    __hash__ = None

    @classmethod
    def create(cls, components: set[tuple[Path, int]]) -> "Dataset":
        return _MockDataset(components)

    @classmethod
    def get_features(cls) -> list[str]:
        return ["one", "two", "three", "four", "five"]


class Mode(Enum):
    """The mode of the application."""

    Classification = "classification"
    Identification = "identification"
    Testing = "test"

    @property
    def dataset_class(self) -> Type[Dataset]:
        """The features the data for this mode provides."""
        match self:
            case Mode.Classification:
                return ClusterClassificationDataset
            case Mode.Testing:
                return _MockDataset
            case _:
                raise NotImplementedError

    @property
    def features(self) -> list[str]:
        """The features the data for this mode provides."""
        return self.dataset_class.get_features()
