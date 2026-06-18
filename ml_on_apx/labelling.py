from typing import Dict, Set


class Label(str):
    pass


class Labels:
    """Track labels used by a dataset or for model training."""

    def __init__(self, labels: Set[Label]):
        self._data: Dict[Label, int] = dict()
        temp = []
        for label in labels:
            temp.append(label)
        temp.sort()
        for i in range(len(temp)):
            self._data[temp[i]] = i

    def get_labels(self) -> Dict[Label, int]:
        return self._data

    def __contains__(self, label: Label):
        return label in self._data.keys()

    def __getitem__(self, label: Label) -> int:
        return self._data[label]
