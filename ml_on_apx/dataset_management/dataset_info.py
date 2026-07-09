"""Stores informaiton relating to a dataset."""

from pathlib import Path
from typing import Iterable

from ml_on_apx.dataset_management import _DS_INFO
from ml_on_apx.labelling import Label, Labels
from ml_on_apx.logging import log_call

DATASET_NAME_REGEX = r"[\w]([\w\s-]*[\w-])?"


class DatasetInfo:
    """Stores information relating to a dataset."""

    def __init__(self, labels: Labels, sources: Iterable[tuple[Path, Label]]) -> None:
        """Create a new DatasetInfo object.

        Args:
            labels (Labels): The labels the dataset uses.
            sources (Iterable[Tuple[Path, Label]]): The sources, paired with their
                labels, this dataset uses.

        Raises:
            ValueError: If a label for a source is not in this dataset's labels.

        """
        self._labels = labels
        self._sources = set(sources)
        for source in self._sources:
            if source[1] not in labels:
                raise ValueError(f"Label {source[1]} not found in provided labels!")

    @log_call(action_type="get_labels" > _DS_INFO)
    def get_labels(self) -> Labels:
        """Get the labels this dataset uses."""
        return self._labels

    @log_call(action_type="get_sources" > _DS_INFO)
    def get_sources(self) -> set[Path]:
        """Get the unlabled sources this dataset uses."""
        return {x[0] for x in self._sources}

    @log_call(action_type="num_source" > _DS_INFO)
    def get_numbered_sources(self) -> set[tuple[Path, int]]:
        """Get the sources this dataset uses with ml-safe integer labels."""
        labled_sources: set[tuple[Path, int]] = set()
        for path, label in self._sources:
            labled_sources.add((path, self._labels[label]))
        return labled_sources

    @log_call(action_type="labeled_sources" > _DS_INFO)
    def get_labeled_sources(self) -> set[tuple[Path, Label]]:
        """Get the sources this dataset uses with human-readable labels."""
        labled_sources: set[tuple[Path, Label]] = set()
        for path, label in self._sources:
            labled_sources.add((path, label))
        return labled_sources

    def __eq__(self, other: object) -> bool:
        """Compare this dataset to another object.

        Args:
            other (object): The other object to compare to.

        Returns:
            bool: True if the other object is a DatasetInfo object with the same
                information.

        """
        if type(other) is not DatasetInfo:
            return False
        return self._labels == other._labels and self._sources == other._sources
