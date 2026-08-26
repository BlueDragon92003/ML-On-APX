"""Modes that this application runs in."""

from enum import Enum
from typing import Type

from ml_on_apx.cluster_classification.cluster_classification_dataset import (
    ClusterClassificationDataset,
)
from ml_on_apx.dataset_management.dataset import Dataset


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
            case _:
                raise NotImplementedError

    @property
    def features(self) -> list[str]:
        """The features the data for this mode provides."""
        return self.dataset_class.get_features()
