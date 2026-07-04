"""Human-readable machine learning labels."""

from typing import Dict, Iterable, Iterator


class Label(str):
    """A human-readable machine learning label."""


class Labels:
    """Track labels used by a dataset or for model training."""

    def __init__(self, labels: Iterable[Label]) -> None:
        """Create a labels set.

        Args:
            labels (Iterable[Label]): The labels in this label set.

        """
        self._data: Dict[Label, int] = {}
        temp = []
        for label in labels:
            temp.append(label)
        temp.sort()
        for i in range(len(temp)):
            self._data[temp[i]] = i

    def __iter__(self) -> Iterator[Label]:
        """Iterate over all labels.

        Yields:
            Label: The labels in this object.

        """
        return iter(self._data.keys())

    def __len__(self) -> int:
        """Get the number of labels.

        Returns:
            int: The number of labels.

        """
        return len(self._data)

    def __contains__(self, label: Label) -> bool:
        """Return if the provided label is in this set.

        Args:
            label (Label): The label to check.

        Returns:
            bool: If that label is in this Labels object.

        """
        return label in self._data.keys()

    def __getitem__(self, label: Label) -> int:
        """Get the integer representation of a label in this set.

        Args:
            label (Label): The label to get the representation from.

        Returns:
            int: The ML-safe integer lable associated with this label.

        """
        return self._data[label]

    def __eq__(self, other: object) -> bool:
        """Evaluate if the object is the same as this label set.

        Args:
            other (object): The other object to compare.

        Returns:
            bool: If the other object is also a Labels object and has the same labels
                and integer representations.

        """
        if type(other) is not Labels:
            return False
        for label, value in self._data.items():
            if other[label] != value:
                return False
        return True
