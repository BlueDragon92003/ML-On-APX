from typing import Dict, Iterable, Iterator


class Label(str):
    pass


class Labels:
    """Track labels used by a dataset or for model training."""

    def __init__(self, labels: Iterable[Label]):
        self._data: Dict[Label, int] = dict()
        temp = []
        for label in labels:
            temp.append(label)
        temp.sort()
        for i in range(len(temp)):
            self._data[temp[i]] = i

    def __iter__(self) -> Iterator[Label]:
        return iter(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, label: Label) -> bool:
        return label in self._data.keys()

    def __getitem__(self, label: Label) -> int:
        return self._data[label]

    def __eq__(self, other: object) -> bool:
        if type(other) is not Labels:
            return False
        for label, value in self._data.items():
            if other[label] != value:
                return False
        return True
