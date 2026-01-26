from typing import Tuple, Set

from cluster import ClusterType

class DatasetSubset:
    """
    Represents a particular subset of all available data sources that will be
    under use.
    """

    def __init__(self, bitstring, filename: str = None, data: Set[str] = None):
        """
        Create a new subset with a given bitstring and a (tuple of)
        corresponding filename(s).
        
        Subsets should either be created as constants later in this file or
        by combining other datasets using the or operator.
        """
        self.bitstring = bitstring
        if (data is None):
            if (filename is None):
                raise ValueError("Must provide a filename")
            self.data = { filename }
        else:
            self.data = data

    def __or__(self, other):
        """Combine with another available dataset into a new, larger dataset"""
        bitstring = self.bitstring | other.bitstring
        data = self.data | other.data
        return DatasetSubset(bitstring, data=data)
    
    def get_hex(self) -> str:
        """
        Get a hex representation of which sets are in this dataset. Enables easy
        classification for pickling.
        """
        bitstring = self.bitstring
        string = ""
        for _ in range(8):
            string = hex(bitstring & 0b1111)[2:] + string
            bitstring = bitstring >> 4
        return string
    
    def get_data(self) -> Set[Tuple[str, ClusterType]]:
        """Retern all filenames considered by this data subset."""
        return self.data

    def __len__(self) -> int:
        """
        Get number of elements in the former two lists, for iteration purposes.
        """
        return len(self.data)

DatasetSubset.DOUBLE_ELECTRON = DatasetSubset(
    0b0000_0000_0000_0000_0000_0000_0000_0001,
    "double_electron.h5"
    )
