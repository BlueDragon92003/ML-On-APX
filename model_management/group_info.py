from typing import List, Type
from labelling import Labels
import torch.nn


class Activation:
    """Represents a specific type of activation."""

    def __init__(self, name: str, activation: Type[torch.nn.Module]):
        self.name = name
        self.activation = activation

    def get_name(self) -> str:
        """Gets the human-readable name for this activation."""
        return self.name

    def get_activation(self) -> Type[torch.nn.Module]:
        """Gets the module class to use this activation."""
        return self.activation

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Activation):
            return self.activation == other.activation
        return False

    @staticmethod
    def get_activations(cls) -> List["Activation"]:
        """Returns a static list of activations this application supports."""
        return [
            Activation("ReLU", torch.nn.ReLU),
        ]


class GroupInfo:
    """Stores training data about the group"""

    def __init__(self, labels: Labels, possible_features: List[str]):
        pass

    def enable_feature(self, feature: str):
        """Set a dataset feature to be used for training or testing."""
        pass

    def disable_feature(self, feature: str):
        """Set a dataset feature to not be used for training or testing."""
        pass

    def insert_layer_below(self, layer: int, activation: Activation, size: int = 1):
        """Add a layer below (closer to output) the specified layer."""
        pass

    def insert_layer_above(self, layer: int, activation: Activation, size: int = 1):
        """Add a layer above (closer to input) the specified layer."""
        pass

    def remove_layer(self, layer: int):
        """Remove the specified layer."""
        pass

    def get_layer_size(self, layer: int) -> int:
        """Get the size of the specified layer."""
        pass

    def set_layer_size(self, layer: int, size: int):
        """Set the size of the specified layer."""
        pass

    def change_layer_size(self, layer: int, by: int):
        """Increase or decrease the size of the specified layer."""
        pass

    def get_layer_activation(self, layer: int) -> Activation:
        """Get the activation used by the specified layer."""
        pass

    def set_layer_activation(self, layer: int, activation: Activation):
        """Set the activation used by the specified layer."""
        pass
