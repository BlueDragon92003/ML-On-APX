from typing import Tuple, Set

from cluster import ClusterType

# Represents a particular subset of all valid 
class DatasetSubset:
    def __init__(self, bitstring, filename: str = None, cluster_type: ClusterType = None, data: Tuple[str, ClusterType] = None):
        self.bitstring = bitstring
        if (data is None):
            if ((filename is None) or (cluster_type is None)):
                raise ValueError("Must provide a filename and cluster type")
            self.data = {(filename,cluster_type)}
        else:
            self.data = data

    # Combine with another available dataset into a new, larger dataset
    def __or__(self, other):
        bitstring = self.bitstring | other.bitstring
        data = self.data | other.data
        return DatasetSubset(bitstring, data=data)
    
    # Get a hex representation of which sets are in this dataset. Enables easy
    # classification for pickling.
    def get_hex(self) -> str:
        bitstring = self.bitstring
        string = ""
        for _ in range(8):
            string = hex(bitstring & 0b1111)[2:] + string
            bitstring = bitstring >> 4
        return string
    
    # Get a set of tuples containing filename and cluster type info.
    def get_data(self) -> Set[Tuple[str, ClusterType]]:
        return self.data

    # Get number of elements in the former two lists, for iteration purposes.
    def __len__(self) -> int:
        return len(self.data)

DatasetSubset.DOUBLE_ELECTRON = DatasetSubset(
    0b0000_0000_0000_0000_0000_0000_0000_0001,
    "double_electron", ClusterType.ELECTROMAGNETIC
    )
