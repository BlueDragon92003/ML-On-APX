"""Modes that this application runs in."""

from enum import Enum

from ml_on_apx.cluster_classification.cluster_classification_dataset import (
    ClusterClassificationDataset,
)


class Mode(Enum):
    """The mode of the application."""

    Classification = "classification"
    Identification = "identification"
    Testing = "test"

    @property
    def features(self) -> list[str]:
        """The features the data for this mode provides."""
        match self:
            case Mode.Classification:
                return ClusterClassificationDataset.get_features()
            case _:
                raise NotImplementedError
