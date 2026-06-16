from typing import Dict, Set


class Label(str):
    pass


class Labels:
    """Track labels used by a dataset or for model training."""

    def __init__(self, labels=Set[Label]):
        pass

    def get_labels(self) -> Dict[Label, int]:
        pass
