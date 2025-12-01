from typing import Tuple, Set

from cluster import ClusterType

class DatasetSubset:
    """
    Represents a particular subset of all available data sources that will be
    under use.
    """

    def __init__(self, bitstring, filename: str = None, cluster_type: ClusterType = None, data: Tuple[str, ClusterType] = None):
        """
        Create a new subset with a given bitstring and either a filename and
        cluster type or other data.
        
        Subsets should either be created as constants later in this file or
        by combining other datasets using the or operator.
        """
        self.bitstring = bitstring
        if (data is None):
            if ((filename is None) or (cluster_type is None)):
                raise ValueError("Must provide a filename and cluster type")
            self.data = {(filename,cluster_type)}
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
        """Get a set of tuples containing filename and cluster type info."""
        return self.data

    def __len__(self) -> int:
        """
        Get number of elements in the former two lists, for iteration purposes.
        """
        return len(self.data)

DatasetSubset.DOUBLE_ELECTRON = DatasetSubset(
    0b0000_0000_0000_0000_0000_0000_0000_0001,
    "double_electron", ClusterType.ELECTROMAGNETIC
    )
