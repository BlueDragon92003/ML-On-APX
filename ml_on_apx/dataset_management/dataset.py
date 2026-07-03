"""The abstract class datasets should inherit from."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Set, Tuple

import torch


class Dataset(ABC, torch.utils.data.Dataset):
    """The abstract class datasets should inherit from."""

    @classmethod
    @abstractmethod
    def create(cls, components: Set[Tuple[Path, int]]) -> "Dataset":
        """Generate a dataset using the provided file names and ml labels.

        Arguments:
        components: A set of (str, SignaTypes) that specify the paths to the `.root`
            files this dataset reads from, and the signal type each root file is
            associated with.

        """
        raise NotImplementedError
