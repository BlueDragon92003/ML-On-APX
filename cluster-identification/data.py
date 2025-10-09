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


def get_data(batch_size, ):
    logging.info("Getting data")
    return DataLoader(load_data(), batch_size=batch_size, shuffle=True)

def load_data():
    pass
