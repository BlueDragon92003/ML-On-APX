import pickle

import logging

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import h5py

from enum import Enum

from cluster import ClusterType

'''
data.py

This file processes data provided in the h5 format into a DataLoader usable by
PyTorch.
'''

class DatasourceType(Enum):
    TRAINING = 1
    TESTING = 2

# Represents a particular subset of all valid 
class DatasetSubset:
    def __init__(self, number, filename, type):
        pass

    # Combine with another available dataset into a new, larger dataset
    def __or__(self, other):
        pass
    
    # Get a hex representation of which sets are in this dataset. Enables easy
    # classification for pickling.
    def get_hex(self):
        pass
    
    # Get the list of names of .h5 files to load.
    def get_filenames(self):
        pass

    # Get the list of cluster types. Indices match those from `get_filenames`. 
    def get_cluster_types(self):
        pass

    # Get number of elements in the former two lists, for iteration purposes.
    def __len__(self):
        pass
DatasetSubset.DOUBLE_ELECTRON = DatasetSubset(
    0b0000_0000_0000_0000_0000_0000_0000_0001,
    "double_electron", ClusterType.ELECTROMAGNETIC
    )

# Return a PyTorch DataLoader with a specified batch size and datatype.
def get_data(datasource_type, batch_size,):
    logging.info("Getting data")
    data = load_data(datasource_type, DatasetSubset.DOUBLE_ELECTRON)
    return DataLoader(
            # The data to load
            data,
            # Set the batch size
            batch_size=batch_size,
            # Provide the data in a random order. This improves training by
            #   giving the optimizer a more hollistic overview of the data,
            #   instead of unintentionally training & optimizing the model on
            #   one class, then the next class, then the next one, etc.  
            shuffle=True,
            )

# Loads the data as a PyTorch Dataset, provided a specific type (training or
# testing) and a set of h5 data to use as samples.
# 
# First, the function checks if such a Dataset has already been created. If so,
# it checks if the preserved Dataset needs to be updated (i.e. at least one of
# the datafiles is newer than the picked Dataset). If the preserved Dataset
# exists and is not outdated, it is unpicked and returned.
# 
# If the preserved dataset is outdated or does not exist, then a new one is
# created from h5 data.
def load_data(datasource_type, datasets):
    pass
    # return ClusterClassificationDataset(datasource_type, data)

class ClusterClassificationDataset(Dataset):
    # Setup the dataset with the provided source type (training or testing) and
    # loaded h5 data file.
    # 
    # The data must be provided in a list of tupels. The first element in the 
    # tuple is the cluster type the datafile provides. The second element is
    # the data itself, loaded from a .h5 file, with the following required
    # structure:
    # 
    def __init__(self, datasource_type, *marked_h5_data):
        pass

    # Overwrites the Dataset method. Provides a tuple of tensors to be given
    # to the model for training or testing purposes. 
    # 
    # The 
    def __getitem__(self,index):
        pass

    # Overwrites the Dataset method. Provides an indication of how many elements
    # are in this dataset.
    def __len__(self):
        pass
