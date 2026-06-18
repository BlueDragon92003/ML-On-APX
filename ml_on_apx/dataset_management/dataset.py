from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Set
import torch


class Dataset(ABC, torch.utils.data.Dataset):
    @abstractmethod
    @classmethod
    def create(cls, components: Set[Tuple[Path, int]]) -> "Dataset":
        pass
