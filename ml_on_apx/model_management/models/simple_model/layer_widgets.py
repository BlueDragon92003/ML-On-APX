"""Widgets for the SimpleModel screen."""

import re
from abc import ABCMeta, abstractmethod
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalGroup
from textual.message import Message
from textual.message_pump import _MessagePumpMeta
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Input, ListItem, ListView
from textual.widgets import Label as TuiLabel

from ml_on_apx.labelling import Label
from ml_on_apx.logging import log_call
from ml_on_apx.model_management.group_info import Activation
from ml_on_apx.model_management.models.simple_model import _SIMPLE
from ml_on_apx.tui_common.get_string_question import GetStringQuestion
from ml_on_apx.tui_common.list_multiselect_question import ListMultiselectQuestion
from ml_on_apx.tui_common.list_select_question import ListSelectQuestion

_WIDGET = "widget" @ _SIMPLE
_SELECTOR = "selector" @ _SIMPLE

_HIDDEN = "hidden" @ _WIDGET
_HIDDEN_SIZE = "size" @ _HIDDEN
_CALLBACK_HIDDEN_SIZE = "callback" > _HIDDEN_SIZE
_HIDDEN_ACTIVATION = "activation" @ _HIDDEN
_CALLBACK_HIDDEN_ACTIVATION = "callback" > _HIDDEN_ACTIVATION

_OUTPUT = "output" @ _WIDGET
_OUTPUT_INCREASE_SIZE = "inc" @ _OUTPUT
_CALLBACK_OUTPUT_INCREASE_SIZE = "callback" > _OUTPUT_INCREASE_SIZE
_OUTPUT_DECREASE_SIZE = "dec" @ _OUTPUT
_CALLBACK_OUTPUT_DECREASE_SIZE = "callback" > _OUTPUT_DECREASE_SIZE
_OUTPUT_SET_SIZE = "size" @ _OUTPUT
_CALLBACK_OUTPUT_SET_SIZE = "callback" > _OUTPUT_SET_SIZE
_OUTPUT_ACTIVATION = "activation" @ _OUTPUT
_CALLBACK_OUTPUT_ACTIVATION = "callback" > _OUTPUT_ACTIVATION

_INPUT = "input" @ _WIDGET
_INPUT_INCREASE_SIZE = "inc" @ _INPUT
_CALLBACK_INPUT_INCREASE_SIZE = "callback" > _INPUT_INCREASE_SIZE
_INPUT_DECREASE_SIZE = "dec" @ _INPUT
_CALLBACK_INPUT_DECREASE_SIZE = "callback" > _INPUT_DECREASE_SIZE
_INPUT_SET_SIZE = "size" @ _INPUT
_CALLBACK_INPUT_SET_SIZE = "callback" > _INPUT_SET_SIZE

_SELECTOR_DEL_LABEL = "del_label" @ _SELECTOR
_CALLBACK_SELECTOR_DEL_LABEL = "callback" > _SELECTOR_DEL_LABEL


class _LayerWidgetMeta(_MessagePumpMeta, ABCMeta):
    pass


class LayerWidget(VerticalGroup, can_focus=True, metaclass=_LayerWidgetMeta):
    """The base class for Layer Widgets."""

    class AddLayerMessage(Message):
        """Message instructing to add a layer below this one."""

        def __init__(self, layer: "LayerWidget", above: bool) -> None:
            """Create a new message."""
            super(LayerWidget.AddLayerMessage, self).__init__()
            self.layer = layer
            self.above = above

    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("+", "increase_size", "Increase layer size"),
        ("-", "decrease_size", "Decrease layer size"),
        ("space", "set_size", "Set size."),
        ("a", "set_activation", "Set activation."),
        ("backspace", "delete_layer", "Delete layer"),
        ("shift+up", "add_above", "New layer above"),
        ("shift+down", "add_below", "New layer below"),
        ("up", "move_up", "New layer above"),
        ("down", "move_down", "New layer below"),
    ]

    layer_size: reactive[int] = reactive(1)
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

    def watch_layer_size(self, new_val: int) -> None:
        """Ensure internal size data is consistent with the display."""
        self.get_child_by_id(
            f"{self.id}-size", expect_type=TuiLabel
        ).content = f"Size: {new_val}"

    def watch_activation(self, new_val: str) -> None:
        """Ensure internal activation data is consistent with the display."""
        self.get_child_by_id(
            f"{self.id}-activation", expect_type=TuiLabel
        ).content = f"Activation: {new_val}"

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

    @abstractmethod
    def action_delete_layer(self) -> None:
        """Delete this layer."""
        raise NotImplementedError

    @log_call(
        action_type="add_above" > _WIDGET, include_args=None, include_result=False
    )
    def action_add_above(self) -> None:
        """Add a new layer above this one."""
        self.post_message(self.AddLayerMessage(self, True))

    @log_call(
        action_type="add_below" > _WIDGET, include_args=None, include_result=False
    )
    def action_add_below(self) -> None:
        """Add a new layer above this one."""
        self.post_message(self.AddLayerMessage(self, False))

    @log_call(action_type="mv_up" > _WIDGET, include_args=None, include_result=False)
    def action_move_up(self) -> None:
        """Add a new layer above this one."""
        self.screen.focus_previous()

    @log_call(action_type="mv_down" > _WIDGET, include_args=None, include_result=False)
    def action_move_down(self) -> None:
        """Add a new layer above this one."""
        self.screen.focus_next()


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
                self.layer_size = int(size)

        self.app.push_screen(
            GetStringQuestion(title="Enter a new layer size:", validator=validator),
            callback=callback_set_size,
        )

    @log_call(action_type=str(_HIDDEN_ACTIVATION))
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

    @log_call(action_type="del" > _HIDDEN, include_args=None, include_result=False)
    def action_delete_layer(self) -> None:
        """Delete this layer."""
        self.remove()


class OutputLayerWidget(LayerWidget):
    """A widget that represents the output layer."""

    labels: reactive[list[Label]] = reactive([])

    def watch_labels(self, new_val: list[Label]) -> None:
        """Ensure that labels and size are kept in sync."""
        self.layer_size = len(new_val)

    @log_call(
        action_type=str(_OUTPUT_INCREASE_SIZE), include_args=None, include_result=False
    )
    def action_increase_size(self) -> None:
        """Increase the size of the layer."""

        @log_call(
            action_type=str(_CALLBACK_OUTPUT_INCREASE_SIZE),
            include_result=False,
        )
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

    @log_call(
        action_type=str(_OUTPUT_DECREASE_SIZE), include_args=None, include_result=False
    )
    def action_decrease_size(self) -> None:
        """Decrease the size of the layer."""

        @log_call(
            action_type=str(_CALLBACK_OUTPUT_DECREASE_SIZE),
            include_result=False,
        )
        def callback_decrease_size(selection: int | None) -> None:
            if selection is not None:
                self.labels = self.labels[:selection] + self.labels[selection + 1 :]

        self.app.push_screen(
            ListSelectQuestion(
                [(self.labels[i], i) for i in range(len(self.labels))],
                title="Select label to remove.",
            )
        )

    @log_call(
        action_type=str(_OUTPUT_SET_SIZE), include_args=None, include_result=False
    )
    def action_set_size(self) -> None:
        """Set the size of the layer."""

        @log_call(
            action_type=str(_CALLBACK_OUTPUT_SET_SIZE),
            include_result=False,
        )
        def callback_set_size(labels: list[Label] | None) -> None:
            if labels is not None:
                self.labels = labels

        self.app.push_screen(LabelSelector(self.labels), callback=callback_set_size)

    @log_call(
        action_type=str(_OUTPUT_ACTIVATION), include_args=None, include_result=False
    )
    def action_set_activation(self) -> None:
        """Set the activation of the layer."""

        @log_call(action_type=_CALLBACK_OUTPUT_ACTIVATION, include_result=False)
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

    @log_call(
        action_type="add_below" > _OUTPUT, include_args=None, include_result=False
    )
    def action_add_below(self) -> None:
        """Do NOT add a layer below this one."""
        self.app.notify("Cannot add a layer below this one.", severity="information")

    @log_call(action_type="del" > _OUTPUT, include_args=None, include_result=False)
    def action_delete_layer(self) -> None:
        """Do NOT delete the output layer."""
        self.app.notify("This layer cannot be deleted.", severity="information")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check to see if an action can be performed."""
        if action in {"add_below", "delete_layer", "move_down"}:
            return False
        return True


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
        self.layer_size = len(new_val)

    @log_call(
        action_type=str(_INPUT_INCREASE_SIZE), include_args=None, include_result=False
    )
    def action_increase_size(self) -> None:
        """Increase the size of the layer."""

        @log_call(action_type=_CALLBACK_INPUT_INCREASE_SIZE, include_result=False)
        def callback_increase_size(selected: str | None) -> None:
            if selected is not None:
                self.features.add(selected)
                self.mutate_reactive(InputLayerWidget.features)

        self.app.push_screen(
            ListSelectQuestion(
                [(x, x) for x in self.all_features if x not in self.features]
            ),
            callback=callback_increase_size,
        )

    @log_call(
        action_type=str(_INPUT_DECREASE_SIZE), include_args=None, include_result=False
    )
    def action_decrease_size(self) -> None:
        """Decrease the size of the layer."""

        @log_call(action_type=_CALLBACK_INPUT_DECREASE_SIZE)
        def callback_decrease_size(selected: str | None) -> None:
            if selected is not None:
                self.features.remove(selected)
                self.mutate_reactive(InputLayerWidget.features)

        self.app.push_screen(
            ListSelectQuestion(
                [(x, x) for x in self.all_features if x in self.features]
            ),
            callback=callback_decrease_size,
        )

    @log_call(action_type=str(_INPUT_SET_SIZE), include_args=None, include_result=False)
    def action_set_size(self) -> None:
        """Set the size of the layer."""

        @log_call(action_type=_CALLBACK_INPUT_SET_SIZE, include_result=False)
        def callback_set_size(selected: set[str] | None) -> None:
            if selected is not None:
                self.features = selected

        self.app.push_screen(
            ListMultiselectQuestion(
                [(x, x, x in self.features) for x in self.all_features],
                title="Select features to use.",
            ),
            callback=callback_set_size,
        )

    @log_call(
        action_type="activation" > _INPUT, include_args=None, include_result=False
    )
    def action_set_activation(self) -> None:
        """Inform the user that the input layer does not have an activation."""
        self.app.notify("Input layer does not have an activation.")

    @log_call(action_type="add_above" > _INPUT, include_args=None, include_result=False)
    def action_add_above(self) -> None:
        """Do NOT add a layer above this one."""
        self.app.notify("Cannot add a layer above this one.", severity="information")

    @log_call(action_type="del" > _INPUT, include_args=None, include_result=False)
    def action_delete_layer(self) -> None:
        """Do NOT delete the output layer."""
        self.app.notify("This layer cannot be deleted.", severity="information")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check to see if an action can be performed."""
        if action in {"add_above", "delete_layer", "move_up"}:
            return False
        return True


class LabelSelector(ModalScreen[list[Label]]):
    """A popup version of the label menu."""

    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("backspace", "delete_label", "Delete selected label"),
        ("escape", "exit", "Exit"),
    ]

    DEFAULT_CLASSES = "Popup"

    labels: reactive[list[Label]] = reactive([])

    def __init__(
        self,
        current: list[Label],
    ) -> None:
        """Initialize a new ListSelectQuestion.

        Args:
            current (list[Label]): A list of options the user can select
                from.

        """
        super().__init__()
        self._labels: list[Label] = current

    def compose(self) -> ComposeResult:
        """Build the widget from its component widgets."""
        with VerticalGroup(classes="container", id="container"):
            yield ListView(id="labels-list", classes="question")
            yield Input(
                placeholder="Label name...",
                id="label-name-input",
            )
            yield Button("New Label", variant="primary", id="label-create-button")
            yield Button("Cancel", variant="default", id="cancel-button")
            yield Button("Save", variant="success", id="save-button")

    def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""
        container = self.get_child_by_id("container")
        container.border_title = "Manage Labels"
        container.border_subtitle = "Press escape to cancel"
        container.get_child_by_id("label-name-input").focus()
        self.labels = self._labels

    @on(Button.Pressed)
    @log_call(action_type="act_button_pressed" > _SELECTOR, include_result=False)
    def handle_button_press(self, message: Button.Pressed) -> None:
        """Handle a button being pressed."""
        match message.button.id:
            case "label-create-button":
                self.create_label()
            case "save-button":
                self.labels.sort()
                self.dismiss(self.labels)
            case "cancel-button":
                self.dismiss(None)

    @on(Input.Submitted)
    @log_call(action_type="act_input_submitted" > _SELECTOR, include_result=False)
    def handle_input_submission(self, message: Input.Submitted) -> None:
        """Handle the Submitted event from an input object.

        Args:
            message (Input.Submitted): The event to handle.

        """
        match message.input.id:
            case "label-name-input":
                self.create_label()

    @log_call(action_type=str(_SELECTOR_DEL_LABEL), include_result=False)
    def action_delete_label(self) -> None:
        """Process the `delete_label` action."""

        @log_call(action_type=_CALLBACK_SELECTOR_DEL_LABEL, include_result=False)
        def delete_label(delete: bool | None) -> None:
            if not delete:
                return
            assert name is not None
            idx = self.labels.index(Label(name))
            self.labels = self.labels[:idx] + self.labels[idx + 1 :]

        labels_list = self.get_widget_by_id("labels-list", expect_type=ListView)
        if labels_list.highlighted_child is not None:
            name = labels_list.highlighted_child.name
            assert name is not None
            idx = self.labels.index(Label(name))
            self.labels = self.labels[:idx] + self.labels[idx + 1 :]

    @log_call(action_type="exit" > _SELECTOR, include_args=None, include_result=False)
    def action_exit(self) -> None:
        """Close without saving."""
        self.dismiss(None)

    def watch_labels(self, old_labels: list[Label], new_labels: list[Label]) -> None:
        """Handle changes in the reactive component `labels`.

        Args:
            old_labels (str | None): The old value for the component.
            new_labels (str | None): The new value for the component.

        """
        self.remake_label_list()

    @log_call(
        action_type="remake_list" > _SELECTOR, include_args=None, include_result=False
    )
    def remake_label_list(self) -> None:
        """Remake and display the list of labels shown to the user."""
        labels_list = self.get_widget_by_id("labels-list", ListView)
        labels_list.clear()
        for label in self.labels:
            labels_list.append(ListItem(TuiLabel(f"{label}"), name=f"{label}"))

    @log_call(
        action_type="create_label" > _SELECTOR, include_args=None, include_result=False
    )
    def create_label(self) -> None:
        """Create a new label from the user string in the view."""
        input = self.get_widget_by_id("label-name-input", Input)
        label_name = input.value
        if not label_name:
            self.app.notify("Please input a label name.")
            input.focus()
            return
        label_name = re.sub(r"\s+", "-", label_name.lower())
        if not re.fullmatch(r"[\w-]+", label_name):
            self.app.notify("Label name is invalid.", severity="error")
            input.focus()
            return
        label = Label(label_name)
        if label in self.labels:
            self.app.notify(f"Label `{label_name}` already exists.", severity="error")
            input.focus()
            return
        # self.labels.append(label)
        self.labels = [*self.labels, label]
        # self.watch_labels()
        input.clear()
        input.focus()
