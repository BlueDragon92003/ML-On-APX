"""A simple, linear sequential ML model info object and generating screen."""

from __future__ import annotations

from typing import ClassVar, cast

import torch
from textual import on
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import HorizontalGroup, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header
from torch import nn

from ml_on_apx.labelling import Labels
from ml_on_apx.logging import log_call
from ml_on_apx.model_management.group_info import Activation, GroupInfo
from ml_on_apx.model_management.model_manager import ModelManager
from ml_on_apx.model_management.models.simple_model import _SIMPLE
from ml_on_apx.model_management.models.simple_model.layer_widgets import (
    HiddenLayerWidget,
    InputLayerWidget,
    LayerWidget,
    OutputLayerWidget,
)

_FEATURE = "feature" @ _SIMPLE
_LAYER = "layer" @ _SIMPLE
_LAYER_ACTIVATION = "activation" @ _LAYER
_LAYER_SIZE = "size" @ _LAYER

_TUI = "tui" @ _SIMPLE


class InputLayerNoActivationError(Exception):
    """Raised when the InputLayer is used where it cannot be."""


class SimpleModel(nn.Module):
    """A machine-learning model."""

    def __init__(self, group_info: SimpleGroupInfo) -> None:
        """Initialize a model."""
        super(SimpleModel, self).__init__()
        activations = Activation.get_activations()
        stack: list[nn.Module] = []
        start_size = group_info.get_layer_size(0)
        for i in range(1, group_info.layer_count):
            end_size = group_info.get_layer_size(i)
            stack.append(nn.Linear(start_size, end_size))
            stack.append(activations[group_info.get_layer_activation(i)].activation())
            start_size = end_size
        self.stack = nn.Sequential(*stack)
        self.mask = torch.tensor(
            [  # TODO test masking system
                group_info.all_features[x] in group_info.features
                for x in range(len(group_info.all_features))
            ],
            dtype=torch.bool,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the forward pass.

        Args:
            x: The input vector for the model to process..

        Returns:
            Tensor: The certainty of the model for each label.

        """
        certainties = self.stack(x[self.mask])
        return certainties


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
    @log_call(action_type="screen" > _SIMPLE, include_result=False)
    def screen(cls, features: list[str]) -> Screen[GroupInfo["SimpleModel"]]:
        """Create a screen to visually create a Simple Model."""
        return SimpleScreen(features)

    @log_call(action_type="screen_preset" > _SIMPLE, include_result=False)
    def screen_with_presets(self) -> Screen[GroupInfo["SimpleModel"]]:
        """Create a screen with presets based on this object."""
        return SimpleScreen(self.all_features, self)

    @log_call(action_type="model" > _SIMPLE, include_result=False)
    def model(self) -> "SimpleModel":
        """Get the simple model this group uses."""
        return SimpleModel(self)

    @log_call(action_type="markdown" > _SIMPLE, include_args=[])
    def get_markdown(self, manager: ModelManager) -> str:
        """Produce the markdown representation of this group."""
        out = "**_Simple model_**\n\n**Inputs**\n"
        for feature in self.all_features:
            if feature not in self.features:
                continue
            out += f"  - {feature}\n"
        out += f"\n**Outputs** ({self._output_activation})\n"
        for label in Labels(*self.labels):
            out += f"  - {label}\n"
        if len(self._hidden_layer_activations) > 0:
            out += "\n**Hidden Layers**\n"
            for size, activation in zip(
                self._hidden_layer_sizes, self._hidden_layer_activations
            ):
                out += f"  - {activation} layer with {size} nodes \n"
        else:
            out += "\nNo Hidden Layers\n"
        return out

    def get_labels(self, manager: ModelManager) -> Labels:
        """Get the labels this group uses."""
        return self._labels

    @property
    def features(self) -> set[str]:
        """The features this group uses."""
        return self._features

    @property
    def all_features(self) -> list[str]:
        """The features available to this group."""
        return self._all_features

    @log_call(action_type="enable" > _FEATURE, include_result=False)
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

    @log_call(action_type="disable" > _FEATURE, include_result=False)
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

    @log_call(action_type="below" > _LAYER, include_result=False)
    def insert_layer(self, activation_name: str, size: int = 1) -> None:
        """Add a layer below (closer to output) the specified layer.

        Args:
            layer (int): The layer index to add below.
            activation_name (str): The activation of the new layer.
            size (int, optional): The size of the layer being added. Defaults to 1.

        """
        self._hidden_layer_sizes.append(size)
        self._hidden_layer_activations.append(activation_name)

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

    @log_call(action_type="set" > _LAYER_ACTIVATION, include_result=False)
    def set_output_activation(self, activation_name: str) -> None:
        """Set the activation used by the specified layer.

        Args:
            activation_name (str): The activation the output layer should use.

        """
        self._output_activation = activation_name

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

    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+s", "save", "Save"),
    ]

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
        self._idinc = 0

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
            yield Button("Cancel.", id="cancel-button")
            yield Button("Save...", id="save-button", variant="primary")

    def on_mount(self) -> None:
        """Finish setup of the widget once it is attached to the DOM."""
        if self._base is not None:
            box = self.get_child_by_id("hidden-layers", expect_type=VerticalScroll)
            input_layer = box.get_child_by_type(InputLayerWidget)
            output_layer = box.get_child_by_type(OutputLayerWidget)
            input_layer.all_features = self._base.all_features
            input_layer.features = self._base.features
            output_layer.labels = list(self._base.labels)
            output_layer.activation = self._base._output_activation
            for hl in range(self._base.layer_count):
                if hl == 0 or hl == self._base.layer_count - 1:
                    continue
                hidden_layer = HiddenLayerWidget(id=f"hl{hl}")
                box.mount(hidden_layer, before="OutputLayerWidget")
                hidden_layer.activation = self._base.get_layer_activation(hl)
                hidden_layer.layer_size = self._base.get_layer_size(hl)

    @on(LayerWidget.AddLayerMessage)
    @log_call(action_type="add_layer" > _TUI, include_result=False)
    def handle_add_layer(self, message: LayerWidget.AddLayerMessage) -> None:
        """Add a new layer above or below the messaging layer."""
        box = self.get_child_by_id("hidden-layers", expect_type=VerticalScroll)
        layer_id = message.layer.id
        new_layer = HiddenLayerWidget(id=f"hl{self._idinc}")
        self._idinc += 1
        if message.above:
            box.mount(new_layer, before=f"#{layer_id}")
        else:
            box.mount(new_layer, after=f"#{layer_id}")
        new_layer.focus()
        message.stop()

    @on(Button.Pressed, "#cancel-button")
    @log_call(action_type="cancel" > _TUI, include_result=False)
    def action_cancel(self, message: Button.Pressed | None = None) -> None:
        """Handle cancelling the creation of the GroupInfo."""
        self.dismiss(None)
        if message is not None:
            message.stop()

    @on(Button.Pressed, "#save-button")
    @log_call(action_type="save" > _TUI, include_result=False)
    def action_save(self, message: Button.Pressed | None = None) -> None:
        """Handle saving the GroupInfo."""
        new_group_info = SimpleGroupInfo(self._features)
        box = self.get_child_by_id("hidden-layers", expect_type=VerticalScroll)
        for child in box.children:
            match child.id:
                case "input":
                    input_layer = cast(InputLayerWidget, child)
                    if len(input_layer.features) < 1:
                        self.notify("Input layer missing features.", severity="error")
                        return
                    for feature in input_layer.features:
                        new_group_info.enable_feature(feature)
                case "output":
                    output_layer = cast(OutputLayerWidget, child)
                    activation = output_layer.activation
                    labels = output_layer.labels
                    if activation is None:
                        self.notify(
                            "Output layer missing activation.", severity="error"
                        )
                        return
                    new_group_info.set_output_activation(activation)
                    if len(labels) < 1:
                        self.notify("Output layer missing labels.", severity="error")
                        return
                    new_group_info.labels = Labels(*labels)
                case _:
                    hidden_layer = cast(HiddenLayerWidget, child)
                    size = hidden_layer.layer_size
                    activation = hidden_layer.activation
                    if activation is None:
                        self.notify("Layer missing activation.", severity="error")
                        return
                    new_group_info.insert_layer(activation, size)
        if message is not None:
            message.stop()
        self.dismiss(new_group_info)
