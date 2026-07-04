"""Information on a model group, including activations."""

from typing import List, Type

import torch.nn

from ml_on_apx.labelling import Labels


class Activation:
    """Represents a specific type of activation."""

    def __init__(self, name: str, activation: Type[torch.nn.Module]) -> None:
        """Create a new Activation.

        Args:
            name (str): The name for the activation.
            activation (Type[torch.nn.Module]): The activation.

        """
        self.name = name
        self.activation = activation

    def get_name(self) -> str:
        """Get the human-readable name for this activation."""
        return self.name

    def get_activation(self) -> Type[torch.nn.Module]:
        """Get the module class to use this activation."""
        return self.activation

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
    def get_activations(cls: Type["Activation"]) -> List["Activation"]:
        """Return a static list of activations this application supports."""
        return [
            Activation("ReLU", torch.nn.ReLU),
        ]


class GroupInfo:
    """Stores training data about the group."""

    def __init__(self, labels: Labels, possible_features: List[str]) -> None:
        """Create a new group info object.

        Args:
            labels (Labels): The labels this group uses.
            possible_features (List[str]): The list of features this group may use as
                model input.

        """

    def enable_feature(self, feature: str) -> None:
        """Set a dataset feature to be used for training or testing.

        Args:
            feature (str): The name of the feature to enable.

        """

    def disable_feature(self, feature: str) -> None:
        """Remove a dataset feature to be used for training or testing.

        Args:
            feature (str): The name of the feature to disable.

        """

    def insert_layer_below(
        self, layer: int, activation: Activation, size: int = 1
    ) -> None:
        """Add a layer below (closer to output) the specified layer.

        Args:
            layer (int): The layer index to add below.
            activation (Activation): The activation of the new layer.
            size (int, optional): The size of the layer being added. Defaults to 1.

        """

    def insert_layer_above(
        self, layer: int, activation: Activation, size: int = 1
    ) -> None:
        """Add a layer above (closer to input) the specified layer.

        Args:
            layer (int): The layer index to add above.
            activation (Activation): The activation the new layer should use.
            size (int, optional): The size of the layer being added. Defaults to 1.

        """

    def remove_layer(self, layer: int) -> None:
        """Remove the specified layer.

        Args:
            layer (int): The layer index to remove.

        """

    def get_layer_size(self, layer: int) -> int:
        """Get the size of the specified layer.

        Args:
            layer (int): The layer index to get the size of.

        Returns:
            int: The size of the layer.

        """

    def set_layer_size(self, layer: int, size: int) -> None:
        """Set the size of the specified layer.

        Args:
            layer (int): The layer index to set the size for.
            size (int): The size of the layer to set.

        """

    def change_layer_size(self, layer: int, by: int) -> None:
        """Increase or decrease the size of the specified layer.

        Args:
            layer (int): The layer index to modify the size of.
            by (int): The number to add to the size of the layer.

        """

    def get_layer_activation(self, layer: int) -> Activation:
        """Get the activation used by the specified layer.

        Args:
            layer (int): The layer index to get the activation from.

        Returns:
            Activation: The activation of that layer.

        """

    def set_layer_activation(self, layer: int, activation: Activation) -> None:
        """Set the activation used by the specified layer.

        Args:
            layer (int): The layer index to set the activation of.
            activation (Activation): The activation the layer should use.

        """
