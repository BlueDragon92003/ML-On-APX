"""Information on a model group, including activations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Self, Type, TypeVar

import torch.nn
from textual.screen import Screen
from torch import nn

from ml_on_apx.logging import log_call
from ml_on_apx.model_management import _MODEL

if TYPE_CHECKING:
    from ml_on_apx.model_management.model_manager import ModelManager

_GROUP_INFO = "group" @ _MODEL
_ACTIVATION = "activation" @ _MODEL


class Activation:
    """Represents a specific type of activation."""

    def __init__(self, name: str, activation: Type[torch.nn.Module]) -> None:
        """Create a new Activation.

        Args:
            name (str): The name for the activation.
            activation (Type[torch.nn.Module]): The activation.

        """
        self._name = name
        self._activation = activation

    @property
    def name(self) -> str:
        """The human-readable name for this activation."""
        return self._name

    @property
    def activation(self) -> Type[torch.nn.Module]:
        """Get the module class to use this activation."""
        return self._activation

    def __eq__(self, other: object) -> bool:
        """Compare this activation to another object.

        Args:
            other (object): The object to compare to.

        Returns:
            bool: True, if the other object is the same type of activation.

        """
        if isinstance(other, Activation):
            return self.activation == other.activation
        return False

    @staticmethod
    @log_call(action_type="list" > _ACTIVATION)
    def get_activations() -> dict[str, "Activation"]:
        """Return a static list of activations this application supports."""
        return {
            x: Activation(x, y)
            for x, y in [
                ("ReLU", torch.nn.ReLU),
                ("Sigmoid", torch.nn.Sigmoid),
                ("Tanh", torch.nn.Tanh),
                ("ELU", torch.nn.ELU),
                ("Hardshrink", torch.nn.Hardshrink),
                ("Hardsigmoid", torch.nn.Hardsigmoid),
                ("Hardtanh", torch.nn.Hardtanh),
                ("Hardswish", torch.nn.Hardswish),
                ("LeakyReLU", torch.nn.LeakyReLU),
                ("LogSigmoid", torch.nn.LogSigmoid),
                ("MultiheadAttention", torch.nn.MultiheadAttention),
                ("PReLU", torch.nn.PReLU),
                ("ReLU6", torch.nn.ReLU6),
                ("RReLU", torch.nn.RReLU),
                ("SELU", torch.nn.SELU),
                ("CELU", torch.nn.CELU),
                ("GELU", torch.nn.GELU),
                ("SiLU", torch.nn.SiLU),
                ("Mish", torch.nn.Mish),
                ("Softplus", torch.nn.Softplus),
                ("Softshrink", torch.nn.Softshrink),
                ("Softsign", torch.nn.Softsign),
                ("Tanhshrink", torch.nn.Tanhshrink),
                ("Threshold", torch.nn.Threshold),
                ("GLU", torch.nn.GLU),
            ]
        }


T = TypeVar("T", bound=nn.Module)


class GroupInfo(ABC, Generic[T]):
    """Stores training data about the group."""

    @classmethod
    @abstractmethod
    def screen(cls) -> Screen[Self]:
        """Create a screen to visually create a GroupInfo object."""
        raise NotImplementedError

    @abstractmethod
    def model(self) -> T:
        """Generate the ML model this group uses."""
        raise NotImplementedError

    @abstractmethod
    def get_markdown(self, manager: ModelManager) -> str:
        """Generate the ML model this group uses."""
        raise NotImplementedError
