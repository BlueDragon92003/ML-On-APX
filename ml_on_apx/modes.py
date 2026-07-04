"""Modes that this application runs in."""

from enum import Enum


class Mode(Enum):
    """The mode of the application."""

    Classification = "classification"
    Identification = "identification"
    Testing = "test"
