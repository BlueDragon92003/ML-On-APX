"""Programmatically manage machine learning datasets."""

from ml_on_apx.logging import Namespace

_DATASET_MANAGEMENT = Namespace("data")
_TUI = "app" @ _DATASET_MANAGEMENT
_APP = "app" @ _TUI
_DS_INFO = "info" @ _DATASET_MANAGEMENT
_MANAGER = "manager" @ _DATASET_MANAGEMENT
_TREE = "tree" @ _DATASET_MANAGEMENT
