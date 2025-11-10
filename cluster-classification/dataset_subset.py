from cluster import ClusterType

# Represents a particular subset of all valid 
class DatasetSubset:
    def __init__(self, bitstring, filename=None, cluster_type=None, data=None):
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
        data = self.filenames | other.filenames
        return DatasetSubset(bitstring, data)
    
    # Get a hex representation of which sets are in this dataset. Enables easy
    # classification for pickling.
    def get_hex(self):
        bitstring = self.bitstring
        string = ""
        for _ in range(8):
            string = hex(bitstring & 0b1111)[2:] + string
            bitstring = bitstring >> 4
        return string
    
    # Get a set of tuples containing filename and cluster type info.
    def get_data(self):
        return self.data.map

    # Get number of elements in the former two lists, for iteration purposes.
    def __len__(self):
        return len(self.data)

DatasetSubset.DOUBLE_ELECTRON = DatasetSubset(
    0b0000_0000_0000_0000_0000_0000_0000_0001,
    ["double_electron"], [ClusterType.ELECTROMAGNETIC]
    )
