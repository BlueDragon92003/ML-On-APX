"""Information on a model group, including activations."""

from typing import Type

import torch.nn

from ml_on_apx.labelling import Labels
from ml_on_apx.logging import log_call
from ml_on_apx.model_management import _MODEL

_GROUP_INFO = "group" @ _MODEL
_ACTIVATION = "activation" @ _MODEL

_FEATURE = "feature" @ _GROUP_INFO
_LAYER = "layer" @ _GROUP_INFO
_LAYER_ACTIVATION = "activation" @ _LAYER
_LAYER_SIZE = "size" @ _LAYER


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


class GroupInfo:
    """Stores training data about the group."""

    DEFAULT_ACTIVATION = "ReLU"

    def __init__(self, labels: Labels, possible_features: set[str]) -> None:
        """Create a new group info object.

        Args:
            labels (Labels): The labels this group uses.
            possible_features (List[str]): The list of features this group may use as
                model input.

        """
        self._labels = labels
        self._input_layer_size = 0
        self._hidden_layer_sizes: list[int] = []
        self._hidden_layer_activations: list[str] = []
        self._output_activation: str = self.DEFAULT_ACTIVATION
        self._output_layer_size = len(labels)
        self._features: set = set()
        self._all_features = possible_features

    @property
    def features(self) -> set[str]:
        """The features this group uses."""
        return self._features

    @property
    def all_features(self) -> set[str]:
        """The features available to this group."""
        return self._all_features

    @log_call(action_type="enable" > _FEATURE)
    def enable_feature(self, feature: str) -> None:
        """Set a dataset feature to be used for training or testing.

        Args:
            feature (str): The name of the feature to enable.

        Raises:
            ValueError: If the provided feature is not tracked by this group.

        """
        if feature not in self._all_features:
            raise ValueError("No such feature!")
        self._features.add(feature)
        self._input_layer_size += 1

    @log_call(action_type="disable" > _FEATURE)
    def disable_feature(self, feature: str) -> None:
        """Remove a dataset feature to be used for training or testing.

        Args:
            feature (str): The name of the feature to disable.

        """
        if feature not in self._all_features:
            raise ValueError("No such feature!")
        if feature in self._features:
            self._features.remove(feature)
            self._input_layer_size -= 1

    @log_call(action_type="below" > _LAYER)
    def insert_layer_below(
        self, layer: int, activation_name: str, size: int = 1
    ) -> None:
        """Add a layer below (closer to output) the specified layer.

        Args:
            layer (int): The layer index to add below.
            activation_name (str): The activation of the new layer.
            size (int, optional): The size of the layer being added. Defaults to 1.

        """
        if layer < 0:
            raise IndexError(f"Index {layer} out of bounds!")
        if layer >= len(self._hidden_layer_sizes) + 1:
            raise IndexError("Cannot add below output layer!")
        self._hidden_layer_sizes.insert(layer, size)
        self._hidden_layer_activations.insert(layer, activation_name)

    @log_call(action_type="above" > _LAYER)
    def insert_layer_above(
        self, layer: int, activation_name: str, size: int = 1
    ) -> None:
        """Add a layer above (closer to input) the specified layer.

        Args:
            layer (int): The layer index to add above.
            activation_name (str): The activation the new layer should use.
            size (int, optional): The size of the layer being added. Defaults to 1.

        """
        if layer <= 0:
            raise IndexError("Cannot add above input layer!")
        if layer > len(self._hidden_layer_sizes) + 1:
            raise IndexError(f"Index {layer} out of bounds!")
        self._hidden_layer_sizes.insert(layer - 1, size)
        self._hidden_layer_activations.insert(layer - 1, activation_name)

    @log_call(action_type="del" > _LAYER)
    def remove_layer(self, layer: int) -> None:
        """Remove the specified layer.

        Args:
            layer (int): The layer index to remove.

        Raises:
            IndexError: If the provided index is out of bounds.
            ValueError: If the input or output layers were selected for removal.

        """
        if layer < 0 or layer > len(self._hidden_layer_sizes) + 1:
            raise IndexError(f"Index {layer} out of bounds!")
        if layer == 0:
            raise ValueError("Cannot remove input layer!")
        if layer == len(self._hidden_layer_sizes) + 1:
            raise ValueError("Cannot remove output layer!")
        self._hidden_layer_sizes.pop(layer - 1)
        self._hidden_layer_activations.pop(layer - 1)

    @log_call(action_type="get" > _LAYER_SIZE)
    def get_layer_size(self, layer: int) -> int:
        """Get the size of the specified layer.

        Args:
            layer (int): The layer index to get the size of.

        Raises:
            IndexError: If the provided index is out of bounds.

        Returns:
            int: The size of the layer.

        """
        if layer < 0:
            raise IndexError(f"Index {layer} out of bounds!")
        elif layer == 0:
            return self._input_layer_size  # layer 0 is "input"
        elif layer <= len(self._hidden_layer_sizes):
            return self._hidden_layer_sizes[layer - 1]  # layer 1 is hidden layer 0
        elif layer == len(self._hidden_layer_sizes) + 1:
            return self._output_layer_size
        else:
            raise IndexError(f"Index {layer} out of bounds!")

    @log_call(action_type="set" > _LAYER_SIZE)
    def set_layer_size(self, layer: int, size: int) -> None:
        """Set the size of the specified layer.

        Args:
            layer (int): The layer index to set the size for.
            size (int): The size of the layer to set.

        Raises:
            IndexError: If the provided index is the input layer, the output layer, or
                out of bounds.
            ValueError: If the user tried set a non-positive layer size.

        """
        if layer < 0:
            raise IndexError(f"Index {layer} out of bounds!")
        elif layer == 0:
            # layer 0 is "input"
            raise IndexError(
                "Index 0 is the input layer. \
                Change its size by enabling or disabling features."
            )
        elif layer <= len(self._hidden_layer_sizes):
            if size <= 0:
                raise ValueError("Invalid layer size!")
            self._hidden_layer_sizes[layer - 1] = size  # layer 1 is hidden layer 0
        elif layer == len(self._hidden_layer_sizes) + 1:
            raise IndexError(
                f"Index {layer} is the output layer. \
                Its size was determined by dataset labels."
            )
        else:
            raise IndexError(f"Index {layer} out of bounds!")

    @log_call(action_type="delta" > _LAYER_SIZE)
    def change_layer_size(self, layer: int, by: int) -> None:
        """Increase or decrease the size of the specified layer.

        Args:
            layer (int): The layer index to modify the size of.
            by (int): The number to add to the size of the layer.

        Raises:
            IndexError: If the provided index is the input layer, the output layer, or
                out of bounds.
            ValueError: If the user tried to change the size of the input or output
                layer.

        """
        if layer < 0:
            raise IndexError(f"Index {layer} out of bounds!")
        elif layer == 0:
            # layer 0 is "input"
            raise IndexError(
                "Index 0 is the input layer. \
                Change its size by enabling or disabling features."
            )
        elif layer <= len(self._hidden_layer_sizes):
            if (new := self._hidden_layer_sizes[layer - 1] + by) <= 0:
                raise ValueError(f"Underset layer size by {by} to {new}!")
            self._hidden_layer_sizes[layer - 1] += by  # layer 1 is hidden layer 0
        elif layer == len(self._hidden_layer_sizes) + 1:
            raise IndexError(
                f"Index {layer} is the output layer. \
                Its size was determined by dataset labels."
            )
        else:
            raise IndexError(f"Index {layer} out of bounds!")

    @log_call(action_type="get" > _LAYER_ACTIVATION)
    def get_layer_activation(self, layer: int) -> str:
        """Get the activation used by the specified layer.

        Args:
            layer (int): The layer index to get the activation from.

        Raises:
            IndexError: If the provided index is out of bounds.
            ValueError: If the user tried to access the activation of the input layer.

        Returns:
            Activation: The activation of that layer.

        """
        if layer < 0:
            raise IndexError(f"Index {layer} out of bounds!")
        elif layer == 0:
            raise ValueError("Input layer does not have an activation!")
        elif layer <= len(self._hidden_layer_sizes):
            return self._hidden_layer_activations[
                layer - 1
            ]  # layer 1 is hidden layer 0
        elif layer == len(self._hidden_layer_sizes) + 1:
            return self._output_activation
        else:
            raise IndexError(f"Index {layer} out of bounds!")

    @log_call(action_type="set" > _LAYER_ACTIVATION)
    def set_layer_activation(self, layer: int, activation_name: str) -> None:
        """Set the activation used by the specified layer.

        Args:
            layer (int): The layer index to set the activation of.
            activation_name (str): The activation the layer should use.

        Raises:
            IndexError: If the provided index is out of bounds.
            ValueError: If the user tried to access the activation of the input layer.

        """
        if layer < 0:
            raise IndexError(f"Index {layer} out of bounds!")
        elif layer == 0:
            raise ValueError("Input layer does not have an activation!")
        elif layer <= len(self._hidden_layer_sizes):
            # layer 1 is hidden layer 0
            self._hidden_layer_activations[layer - 1] = activation_name
        elif layer == len(self._hidden_layer_sizes) + 1:
            self._output_activation = activation_name
        else:
            raise IndexError(f"Index {layer} out of bounds!")

    @property
    def labels(self: "GroupInfo") -> Labels:
        """The labels models in this group train on."""
        return self._labels

    @property
    def layer_count(self) -> int:
        """The number of layers for the models in this group."""
        return len(self._hidden_layer_sizes) + 2
