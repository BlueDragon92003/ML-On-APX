import pickle

from typing import Tuple

import logging

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import h5py

from enum import Enum

from cluster import ClusterType
from dataset_subset import DatasetSubset

'''
data.py

This file processes data provided in the h5 format into a DataLoader usable by
PyTorch.
'''

class DatasourceType(Enum):
    TRAINING = 1
    TESTING = 2

# Return a PyTorch DataLoader with a specified batch size and datatype.
def get_data(datasource_type: DatasourceType, batch_size: int,):
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
def load_data(datasource_type: DatasourceType, datasets: DatasetSubset):
    dataset_code = datasets.get_hex()
    # return ClusterClassificationDataset(datasource_type, data)

class ClusterClassificationDataset(Dataset):
    # Setup the dataset with the provided source type (training or testing) and
    # loaded h5 data file.
    # 
    # The data must be provided in a list of tuples. The first element in the 
    # tuple is the cluster type the datafile provides. The second element is
    # the data itself, loaded from a .h5 file, with the following required
    # structure:
    # 
    def __init__(self, datasource_type: DatasourceType, *marked_h5_data: h5py.File):
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
