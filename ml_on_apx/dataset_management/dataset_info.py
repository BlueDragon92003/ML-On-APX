from ml_on_apx.labelling import Label, Labels
from pathlib import Path
from typing import Tuple, Set, Iterable

DATASET_NAME_REGEX = r"[\w]([\w\s-]*[\w-])?"


class DatasetInfo:
    """Stores information relating to a dataset."""

    def __init__(self, labels: Labels, sources: Iterable[Tuple[Path, Label]]):
        self._labels = labels
        self._sources = set(sources)
        for source in self._sources:
            if source[1] not in labels:
                raise ValueError(f"Label {source[1]} not found in provided labels!")

    def get_labels(self) -> Labels:
        return self._labels

    def get_sources(self) -> Set[Path]:
        return set(map(lambda x: x[0], self._sources))

    def get_numbered_sources(self) -> Set[Tuple[Path, int]]:
        labled_sources: Set[Tuple[Path, int]] = set()
        for path, label in self._sources:
            labled_sources.add((path, self._labels[label]))
        return labled_sources

    def get_labeled_sources(self) -> Set[Tuple[Path, Label]]:
        labled_sources: Set[Tuple[Path, Label]] = set()
        for path, label in self._sources:
            labled_sources.add((path, label))
        return labled_sources

    def __eq__(self, other: object) -> bool:
        if type(other) is not DatasetInfo:
            return False
        return self._labels == other._labels and self._sources == other._sources
