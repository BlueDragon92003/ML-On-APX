from cluster import ClusterType

# Represents a particular subset of all valid 
class DatasetSubset:
    def __init__(self, number, filenames, cluster_types):
        self.bitstring = number
        self.filenames = filenames
        self.cluster_types = cluster_types

    # Combine with another available dataset into a new, larger dataset
    def __or__(self, other):
        bitstring = self.bitstring | other.bitstring
        filenames = self.filenames + other.filenames
        cluster_types = self.cluster_types + self.cluster_types
        return DatasetSubset(bitstring, filenames, cluster_types)
    
    # Get a hex representation of which sets are in this dataset. Enables easy
    # classification for pickling.
    def get_hex(self):
        bitstring = self.bitstring
        string = ""
        for _ in range(8):
            string = hex(bitstring & 0b1111)[2:] + string
            bitstring = bitstring >> 4
        return string
    
    # Get the list of names of .h5 files to load.
    def get_filenames(self):
        return self.filenames

    # Get the list of cluster types. Indices match those from `get_filenames`. 
    def get_cluster_types(self):
        return self.cluster_types

    # Get number of elements in the former two lists, for iteration purposes.
    def __len__(self):
        return len(self.cluster_types)

DatasetSubset.DOUBLE_ELECTRON = DatasetSubset(
    0b0000_0000_0000_0000_0000_0000_0000_0001,
    ["double_electron"], [ClusterType.ELECTROMAGNETIC]
    )
