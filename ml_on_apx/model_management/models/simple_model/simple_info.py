"""A simple, linear sequential ML model info object and generating screen."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from textual import on
from textual.app import ComposeResult
from textual.containers import HorizontalGroup, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header

from ml_on_apx.labelling import Labels
from ml_on_apx.logging import log_call
from ml_on_apx.model_management.group_info import Activation, GroupInfo
from ml_on_apx.model_management.model_manager import ModelManager
from ml_on_apx.model_management.models.simple_model import _SIMPLE
from ml_on_apx.model_management.models.simple_model.layer_widgets import (
    HiddenLayerWidget,
    InputLayerWidget,
    OutputLayerWidget,
)

if TYPE_CHECKING:
    from ml_on_apx.model_management.models.simple_model.simple_model import SimpleModel

_FEATURE = "feature" @ _SIMPLE
_LAYER = "layer" @ _SIMPLE
_LAYER_ACTIVATION = "activation" @ _LAYER
_LAYER_SIZE = "size" @ _LAYER

_TUI = "tui" @ _SIMPLE


class InputLayerNoActivationError(Exception):
    """Raised when the InputLayer is used where it cannot be."""


class InputLayerModificationError(Exception):
    """Raised when the InputLayer is used where it cannot be."""


class OutputLayerModificationError(Exception):
    """Raised when the InputLayer is used where it cannot be."""


class SimpleGroupInfo(GroupInfo["SimpleModel"]):
    """Stores training data about the group."""

    DEFAULT_ACTIVATION = Activation.get_activations()["ReLU"].name
    # `get_activations` call to ensure the default activation exists.

    def __init__(self, possible_features: list[str]) -> None:
        """Create a new group info object.

        Args:
            labels (Labels): The labels this group uses.
            possible_features (List[str]): The list of features this group may use as
                model input.

        """
        self._labels = Labels()
        self._input_layer_size = 0
        self._hidden_layer_sizes: list[int] = []
        self._hidden_layer_activations: list[str] = []
        self._output_activation: str = self.DEFAULT_ACTIVATION
        self._output_layer_size = len(self._labels)
        self._features: set[str] = set()
        self._all_features = possible_features

    @classmethod
    def screen(cls, features: list[str]) -> Screen[GroupInfo["SimpleModel"]]:
        """Create a screen to visually create a Simple Model."""
        return SimpleScreen(features)

    def screen_with_presets(self) -> Screen[GroupInfo["SimpleModel"]]:
        """Create a screen with presets based on this object."""
        return SimpleScreen(self.all_features, self)

    def model(self) -> "SimpleModel":
        """Get the simple model this group uses."""
        return SimpleModel(self)

    @log_call(action_type="markdown" > _SIMPLE)
    def get_markdown(self, manager: ModelManager) -> str:
        """Produce the markdown representation of this group."""

        def format_hidden_layers(zipped: zip[tuple[int, str]]) -> Iterable[str]:
            for zsize, zact in zipped:
                yield f"{zsize} ({zact})"

        def list_print(the_list: Iterable[str]) -> str:
            out = ""
            for li in the_list:
                out += li + ", "
            return out[:-2]

        hl = list_print(
            format_hidden_layers(
                zip(self._hidden_layer_sizes, self._hidden_layer_activations)
            )
        )

        return f"""**_Simple model_**
**Inputs**: {self._input_layer_size} ({list_print(self._features)})
**Outputs**: {list_print(self._labels)} ({self._output_activation})
**Hidden Layers**: {len(self._hidden_layer_sizes)} ({hl})
"""

    @property
    def features(self) -> set[str]:
        """The features this group uses."""
        return self._features

    @property
    def all_features(self) -> list[str]:
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
            raise ValueError()
        self._features.add(feature)
        self._input_layer_size += 1

    @log_call(action_type="disable" > _FEATURE)
    def disable_feature(self, feature: str) -> None:
        """Remove a dataset feature to be used for training or testing.

        Args:
            feature (str): The name of the feature to disable.

        """
        if feature not in self._all_features:
            raise ValueError()
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
            raise IndexError()
        if layer >= len(self._hidden_layer_sizes) + 1:
            raise IndexError()
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
            raise IndexError()
        if layer > len(self._hidden_layer_sizes) + 1:
            raise IndexError()
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
            raise IndexError()
        if layer == 0:
            raise InputLayerModificationError()
        if layer == len(self._hidden_layer_sizes) + 1:
            raise OutputLayerModificationError
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
            raise IndexError(layer)
        elif layer == 0:
            return self._input_layer_size  # layer 0 is "input"
        elif layer <= len(self._hidden_layer_sizes):
            return self._hidden_layer_sizes[layer - 1]  # layer 1 is hidden layer 0
        elif layer == len(self._hidden_layer_sizes) + 1:
            return self._output_layer_size
        else:
            raise IndexError(layer)

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
            raise IndexError()
        elif layer == 0:
            # layer 0 is "input"
            raise InputLayerModificationError()
        elif layer <= len(self._hidden_layer_sizes):
            if size <= 0:
                raise ValueError(size)
            self._hidden_layer_sizes[layer - 1] = size  # layer 1 is hidden layer 0
        elif layer == len(self._hidden_layer_sizes) + 1:
            raise OutputLayerModificationError()
        else:
            raise IndexError()

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
            raise IndexError()
        elif layer == 0:
            # layer 0 is "input"
            raise InputLayerModificationError()
        elif layer <= len(self._hidden_layer_sizes):
            if (new := self._hidden_layer_sizes[layer - 1] + by) <= 0:
                raise ValueError(new)
            self._hidden_layer_sizes[layer - 1] += by  # layer 1 is hidden layer 0
        elif layer == len(self._hidden_layer_sizes) + 1:
            raise OutputLayerModificationError()
        else:
            raise IndexError()

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
            raise IndexError()
        elif layer == 0:
            raise InputLayerNoActivationError
        elif layer <= len(self._hidden_layer_sizes):
            return self._hidden_layer_activations[
                layer - 1
            ]  # layer 1 is hidden layer 0
        elif layer == len(self._hidden_layer_sizes) + 1:
            return self._output_activation
        else:
            raise IndexError()

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
            raise IndexError()
        elif layer == 0:
            raise InputLayerNoActivationError
        elif layer <= len(self._hidden_layer_sizes):
            # layer 1 is hidden layer 0
            self._hidden_layer_activations[layer - 1] = activation_name
        elif layer == len(self._hidden_layer_sizes) + 1:
            self._output_activation = activation_name
        else:
            raise IndexError()

    @property
    def labels(self) -> Labels:
        """The labels models in this group train on."""
        return self._labels

    @labels.setter
    def labels(self, new: Labels) -> None:
        self._labels = new
        self._output_layer_size = len(self._labels)

    @property
    def layer_count(self) -> int:
        """The number of layers for the models in this group."""
        return len(self._hidden_layer_sizes) + 2


class SimpleScreen(Screen[SimpleGroupInfo]):
    """A screen that creates a SimpleGroupInfo object."""

    def __init__(
        self, features: list[str], base: SimpleGroupInfo | None = None
    ) -> None:
        """Create a new screen.

        Args:
            labels (Labels): The labels this group uses.
            features (list[str]): The features this group uses.
            base (SimpleGroupInfo | None, optional): A base GroupInfo to build on or
                alter. Defaults to None.

        """
        super(SimpleScreen, self).__init__()
        self._features = features
        self._base = base

    def compose(self) -> ComposeResult:
        """Build the screen from its component widgets.

        Returns:
            ComposeResult: An iterator of widgets this screen incorporates.

        Yields:
            Iterator[ComposeResult]: A widget to incorporated.

        """
        yield Header()
        yield Footer()
        with VerticalScroll(id="hidden-layers"):
            yield InputLayerWidget(self._features, id="input")
            yield OutputLayerWidget(id="output")
        with HorizontalGroup():
            yield Button("Cancel.", name="cancel")
            yield Button("Save...", name="save", variant="primary")

    def on_mount(self) -> None:
        """Finish setup of the widget once it is attached to the DOM."""
        if self._base is not None:
            box = self.get_child_by_id("hidden-layers", expect_type=VerticalScroll)
            input_layer = box.get_child_by_type(InputLayerWidget)
            output_layer = box.get_child_by_type(OutputLayerWidget)
            input_layer.all_features = self._base.all_features
            input_layer.features = self._base.features
            output_layer.labels = list(self._base.labels)
            for hl in range(self._base.layer_count):
                if hl == 0 or hl == self._base.layer_count - 1:
                    continue
                hidden_layer = HiddenLayerWidget(id=f"hl{hl}")
                hidden_layer.activation = self._base.get_layer_activation(hl)
                hidden_layer.layer_size = self._base.get_layer_size(hl)
                box.mount(hidden_layer, before="OutputLayerWidget")

    @on(Button.Pressed)
    def handle_button_press(self, message: Button.Pressed) -> None:
        """Handle a button press."""
        match message.button.name:
            case "cancel":
                self.dismiss(None)
            case "save":
                # TODO validate everything and return object
                raise NotImplementedError
            case _:
                raise ValueError
        message.stop()
