'''
data.py

This file processes data provided in CSV format into DataLoader usable by
PyTorch.
'''

import logging

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

from enum import Enum

class Type(Enum):
    TRAINING = 1
    TESTING = 2

def get_data(type, batch_size, ):
    logging.info("Getting data")
    match type:
        case TRAINING:
            logging.info("Getting training data")
            return DataLoader(load_data(), batch_size=batch_size, shuffle=True)
        case TESTING:
            logging.info("Getting testing data")
            return DataLoader(load_data(), batch_size=batch_size, shuffle=True)

def load_data():
    pass
