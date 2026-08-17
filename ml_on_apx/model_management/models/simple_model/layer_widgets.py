"""Widgets for the SimpleModel screen."""

import re
from abc import ABCMeta, abstractmethod
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalGroup
from textual.message_pump import _MessagePumpMeta
from textual.reactive import reactive
from textual.widgets import Label as TuiLabel

from ml_on_apx.labelling import Label
from ml_on_apx.logging import log_call
from ml_on_apx.model_management.group_info import Activation
from ml_on_apx.model_management.models.simple_model import _SIMPLE
from ml_on_apx.tui_common.get_string_question import GetStringQuestion
from ml_on_apx.tui_common.list_select_question import ListSelectQuestion

_WIDGET = "widget" @ _SIMPLE

_HIDDEN = "hidden" @ _WIDGET
_HIDDEN_SIZE = "size" @ _HIDDEN
_CALLBACK_HIDDEN_SIZE = "callback" > _HIDDEN_SIZE
_HIDDEN_ACTIVATION = "activation" @ _HIDDEN
_CALLBACK_HIDDEN_ACTIVATION = "callback" > _HIDDEN_ACTIVATION


class _LayerWidgetMeta(_MessagePumpMeta, ABCMeta):
    pass


class LayerWidget(VerticalGroup, can_focus=True, metaclass=_LayerWidgetMeta):
    """The base class for Layer Widgets."""

    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("+", "increase_size", "Increase layer size"),
        ("-", "decrease_size", "Decrease layer size"),
        (" ", "set_size", "Set size."),
        ("a", "set_activation", "Set activation."),
    ]

    size: reactive[int] = reactive(1)
    activation: reactive[str | None] = reactive(None)

    def compose(self) -> ComposeResult:
        """Build the widget from its component widgets.

        Returns:
            ComposeResult: An iterator of widgets this screen incorporates.

        Yields:
            Iterator[ComposeResult]: A widget to incorporated.

        """
        yield TuiLabel("loading", id=f"{self.id}-size")
        yield TuiLabel("loading", id=f"{self.id}-activation")

    def on_mount(self) -> None:
        """Finish setup of the widget once it is attached to the DOM."""
        self.get_child_by_id(
            f"{self.id}-activation", expect_type=TuiLabel
        ).content = f"Activation: {self.activation}"
        self.get_child_by_id(
            f"{self.id}-size", expect_type=TuiLabel
        ).content = f"Size: {self.size}"

    def watch_size(self, new_val: int) -> None:
        """Ensure internal size data is consistent with the display."""
        self.get_child_by_id(
            f"{self.id}-size", expect_type=TuiLabel
        ).content = f"Size: {new_val}"

    def watch_activation(self, new_val: str) -> None:
        """Ensure internal activation data is consistent with the display."""
        self.get_child_by_id(
            f"{self.id}-activation", expect_type=TuiLabel
        ).content = f"Activation: {new_val}"

    # TODO messages for adding above/removing below

    @abstractmethod
    def action_increase_size(self) -> None:
        """Increase the size of the layer."""
        raise NotImplementedError

    @abstractmethod
    def action_decrease_size(self) -> None:
        """Decrease the size of the layer."""
        raise NotImplementedError

    @abstractmethod
    def action_set_size(self) -> None:
        """Set the size of the layer."""
        raise NotImplementedError

    @abstractmethod
    def action_set_activation(self) -> None:
        """Set the activation of the layer."""
        raise NotImplementedError


class HiddenLayerWidget(LayerWidget):
    """A widget that represents a hidden layer."""

    @log_call(action_type="inc" > _HIDDEN)
    def action_increase_size(self) -> None:
        """Increase the size of the layer."""
        self.size += 1

    @log_call(action_type="dec" > _HIDDEN)
    def action_decrease_size(self) -> None:
        """Decrease the size of the layer."""
        self.size -= 1

    @log_call(action_type=str(_HIDDEN_SIZE))
    def action_set_size(self) -> None:
        """Set the size of the layer."""

        def validator(string: str) -> bool:
            try:
                p = int(string)
            except ValueError:
                return False
            else:
                return p > 0

        @log_call(action_type=_CALLBACK_HIDDEN_SIZE)
        def callback_set_size(size: str | None) -> None:
            if size is not None:
                self.size = int(size)

        self.app.push_screen(
            GetStringQuestion(title="Enter a new layer size:", validator=validator),
            callback=callback_set_size,
        )

    @log_call(action_type="activation" > _HIDDEN)
    def action_set_activation(self) -> None:
        """Set the activation of the layer."""

        @log_call(action_type=_CALLBACK_HIDDEN_ACTIVATION)
        def callback_set_activation(activation: str | None) -> None:
            if activation is not None:
                self.activation = activation
            if self.activation is None:
                self.remove()

        self.app.push_screen(
            ListSelectQuestion(
                options=[(x, x) for x in Activation.get_activations().keys()],
                title="Choose the new activation:",
            ),
            callback=callback_set_activation,
        )


# TODO logging
class OutputLayerWidget(LayerWidget):
    """A widget that represents the output layer."""

    labels: reactive[list[Label]] = reactive([])

    def watch_labels(self, new_val: list[Label]) -> None:
        """Ensure that labels and size are kept in sync."""
        self.size = len(new_val)

    def action_increase_size(self) -> None:
        """Increase the size of the layer."""

        def callback_increase_size(input: str | None) -> None:
            label_name = input
            if not label_name:
                self.app.notify("Please input a label name.")
                return
            label_name = re.sub(r"\s+", "-", label_name.lower())
            if not re.fullmatch(r"[\w-]+", label_name):
                self.app.notify("Label name is invalid.", severity="error")
                return
            label = Label(label_name)
            if label in self.labels:
                self.app.notify(
                    f"Label `{label_name}` already exists.", severity="error"
                )
                return
            self.labels = [*self.labels, label]

        self.app.push_screen(
            GetStringQuestion(
                title="Label name",
            ),
            callback=callback_increase_size,
        )

    def action_decrease_size(self) -> None:
        """Decrease the size of the layer."""

        def callback_decrease_size(selection: int | None) -> None:
            if selection is not None:
                self.labels = self.labels[:selection] + self.labels[selection + 1 :]

        self.app.push_screen(
            ListSelectQuestion(
                [(self.labels[i], i) for i in range(len(self.labels))],
                title="Select label to remove.",
            )
        )

    def action_set_size(self) -> None:
        """Set the size of the layer."""
        # TODO

    def action_set_activation(self) -> None:
        """Set the activation of the layer."""

        @log_call(action_type=_CALLBACK_HIDDEN_ACTIVATION)
        def callback_set_activation(activation: str | None) -> None:
            if activation is not None:
                self.activation = activation
            if self.activation is None:
                self.remove()

        self.app.push_screen(
            ListSelectQuestion(
                options=[(x, x) for x in Activation.get_activations().keys()],
                title="Choose the new activation:",
            ),
            callback=callback_set_activation,
        )


# TODO Logging
class InputLayerWidget(LayerWidget):
    """A widget that represents the input layer."""

    features: reactive[set[str]] = reactive(set())

    def __init__(
        self,
        features: list[str],
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        """Create a InputLayerWidget."""
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.all_features = features

    def on_mount(self) -> None:
        """Finish setup of the widget once it is attached to the DOM."""
        super().on_mount()
        self.get_child_by_id(
            f"{self.id}-activation", expect_type=TuiLabel
        ).display = False

    def watch_features(self, new_val: set[str]) -> None:
        """Ensure that selected features and size remain in sync."""
        self.size = len(new_val)

    def action_increase_size(self) -> None:
        """Increase the size of the layer."""

        def callback_increase_size(selected: str | None) -> None:
            if selected is not None:
                self.features.add(selected)

        self.app.push_screen(
            ListSelectQuestion(
                [(x, x) for x in self.all_features if x not in self.features]
            ),
            callback=callback_increase_size,
        )

    def action_decrease_size(self) -> None:
        """Decrease the size of the layer."""

        def callback_decrease_size(selected: str | None) -> None:
            if selected is not None:
                self.features.remove(selected)

        self.app.push_screen(
            ListSelectQuestion(
                [(x, x) for x in self.all_features if x in self.features]
            ),
            callback=callback_decrease_size,
        )

    def action_set_size(self) -> None:
        """Set the size of the layer."""
        # TODO

    def action_set_activation(self) -> None:
        """Inform the user that the input layer does not have an activation."""
        self.app.notify("Input layer does not have an activation.")
