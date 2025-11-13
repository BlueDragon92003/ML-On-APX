from typing import Set, Tuple

from torch.utils.data import Dataset
import h5py

from data import DatasourceType
from cluster import ClusterType

class ClusterClassificationDataset(Dataset):
    # Setup the dataset with the provided source type (training or testing) and
    # loaded h5 data file.
    # 
    # The data must be provided in a list of tuples. The first element in the 
    # tuple is the cluster type the datafile provides. The second element is
    # the data itself, loaded from a .h5 file, with the following required
    # structure:
    # 
    def __init__(self, datasource_type: DatasourceType, marked_h5_data: Set[Tuple[ClusterType, h5py.File]]):
        pass

    # Overwrites the Dataset method. Provides a tuple of tensors to be given
    # to the model for training or testing purposes. 
    # 
    # The 
    def __getitem__(self, index):
        pass

    # Overwrites the Dataset method. Provides an indication of how many elements
    # are in this dataset.
    def __len__(self):
        pass
