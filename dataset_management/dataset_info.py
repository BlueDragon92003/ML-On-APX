from labelling import Label, Labels
from pathlib import Path
from typing import List, Tuple


class DatasetInfo:
    """Stores information relating to a dataset."""

    def __init__(self, labels: Labels):
        pass

    def get_labels(self) -> Labels:
        pass

    def add_sources(self, sources: List[Path], label: Label):
        pass

    def get_labeled_sources(self) -> List[Tuple[Path, Label]]:
        pass

    def remove_source(self, source: Path):
        pass
