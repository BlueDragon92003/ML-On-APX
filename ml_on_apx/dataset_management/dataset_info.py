"""Stores informaiton relating to a dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ml_on_apx.dataset_management import _DS_INFO
from ml_on_apx.labelling import Label, Labels
from ml_on_apx.logging import log_call

if TYPE_CHECKING:
    from ml_on_apx.dataset_management.dataset_manager import DatasetManager


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
        self._sources = frozenset(sources)
        for source in self._sources:
            if source[1] not in labels:
                raise ValueError(source[1])

    @property
    @log_call(action_type="get_labels" > _DS_INFO)
    def labels(self) -> Labels:
        """The labels this dataset uses."""
        return self._labels

    @property
    @log_call(action_type="get_sources" > _DS_INFO)
    def sources(self) -> set[Path]:
        """The unlabled sources this dataset uses."""
        return {x[0] for x in self._sources}

    @property
    @log_call(action_type="num_source" > _DS_INFO)
    def numbered_sources(self) -> set[tuple[Path, int]]:
        """The sources this dataset uses with ml-safe integer labels."""
        labled_sources: set[tuple[Path, int]] = set()
        for path, label in self._sources:
            labled_sources.add((path, self._labels[label]))
        return labled_sources

    @property
    @log_call(action_type="labeled_sources" > _DS_INFO)
    def labeled_sources(self) -> set[tuple[Path, Label]]:
        """The sources this dataset uses with human-readable labels."""
        labled_sources: set[tuple[Path, Label]] = set()
        for path, label in self._sources:
            labled_sources.add((path, label))
        return labled_sources

    @log_call(action_type="markdown" > _DS_INFO)
    def get_markdown(self, manager: DatasetManager) -> str:
        """Produce a markdown summary from a DatasetInfo object.

        Args:
            dsinfo (DatasetInfo): The DatasetInfo object the markdown should be produced
                from.
            manager (DatasetManager): The DatasetManager that owns `dsinfo`

        Returns:
            str: The markdown string for `dsinfo`.

        """
        markdown = "The dataset uses the following labels:\n"
        for label in self.labels:
            markdown += f"  - {label}\n"
        markdown += "\nThe dataset uses the following sources:\n"
        for source in self.labeled_sources:
            markdown += (
                f"  - {source[0].relative_to(manager.root_dir_path)} ({source[1]})\n"
            )
        return markdown

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

    def __hash__(self) -> int:
        """Hash this object."""
        return hash(self._labels) + hash(self._sources)
