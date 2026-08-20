"""The screen for managing models within a group."""

from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import HorizontalGroup, VerticalGroup, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, ListView, Markdown, Rule
from textual.widgets import Label as TuiLabel

from ml_on_apx.logging import log_call
from ml_on_apx.model_management import _TUI
from ml_on_apx.model_management.model_manager import ModelManager

"""
-----------------------------------------------------------------------------
|  < GROUP NAME >   |   < MODEL NAME >                                      |
|  [    Back    ]   |   [ Rename ][ Delete ]                                |
|  < model name >   |   < Model Info ...                                    |
|  < model name >   |                                                       |
|  < model name >   |                    >                                  |
|  < model name >   |   Test Results                                        |
|  < model name >   |   [ Test Model ]                                      |
|  < model name >   |   ▸ < date >                                          |
|  < model name >   |   ▾ < date >                                          |
|  < model name >   |     < test info >                                     |
|  [ Train Model ]  |   ▸ < date >                                          |
-----------------------------------------------------------------------------
"""

_MODEL_VIEW = "model" @ _TUI


class ModelView(Screen[None]):
    """A view for the available models in a group."""

    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("esc", "back", "Back"),
        ("N", "train_model", "Train new model"),
        ("R", "rename_model", "Rename model"),
        ("D", "delete_model", "Delete model"),
        ("T", "test_model", "Test model"),
    ]

    selected_model: reactive[str | None] = reactive(None)

    def __init__(self, group_name: str, manager: ModelManager) -> None:
        """Create a new ModelView."""
        super(ModelView, self).__init__()
        self._group_name = group_name
        self._manager = manager

    def compose(self) -> ComposeResult:
        """Build the screen from its component widgets."""
        yield Header()
        yield Footer()
        with VerticalGroup(id="navigation-panel"):
            yield TuiLabel(self._group_name, id="group-title", variant="primary")
            yield Button("Back", id="return-button")
            yield ListView(id="model-list")
            yield Button("Train New Model", id="new-model-button")
        with VerticalScroll(classes="container", id="model-info-view"):
            yield TuiLabel("", classes="title", id="model-name")
            with HorizontalGroup(id="control-buttons"):
                yield Button("Rename", id="rename-model-button")
                yield Button("Delete", variant="error", id="delete-model-button")
            yield Markdown(id="model-info-box")
            with VerticalGroup(id="test-list"):
                yield Rule()
                yield TuiLabel("Test Results", classes="title", id="tests-title")
                yield Button("Run test", variant="primary", id="run-test-button")

    def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""
        self.remake_model_list()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check to see if an action can be performed."""

    def validate_selected_model(self, val: str | None) -> str | None:
        """Validate new values for the reactive component `selected_model`."""

    def watch_selected_model(self, new_val: str | None) -> None:
        """Handle changes in the reactive component `selected_model`."""

    @on(Button.Pressed, "#return-button")
    @log_call(action_type="back" > _MODEL_VIEW, include_args=[], include_result=False)
    def back_to_group_view(self, message: Button.Pressed | None = None) -> None:
        """Return to the group view."""
        self.dismiss()

    @on(Button.Pressed, "#new-model-button")
    @log_call(action_type="train" > _MODEL_VIEW, include_args=[], include_result=False)
    def train_new_model(self, message: Button.Pressed | None = None) -> None:
        """Set the TrainingJob."""

    @on(Button.Pressed, "#rename-model-button")
    @log_call(action_type="rename" > _MODEL_VIEW, include_args=[], include_result=False)
    def rename_model(self, message: Button.Pressed | None = None) -> None:
        """Rename the current Model."""

    @on(Button.Pressed, "#delete-model-button")
    @log_call(action_type="delete" > _MODEL_VIEW, include_args=[], include_result=False)
    def delete_model(self, message: Button.Pressed | None = None) -> None:
        """Delete the current Model."""

    @on(Button.Pressed, "#run-test-button")
    @log_call(action_type="test" > _MODEL_VIEW, include_args=[], include_result=False)
    def run_test(self, message: Button.Pressed | None = None) -> None:
        """Set the TestingJob."""

    @on(ListView.Selected, "#delete-model-button")
    @log_call(action_type="select" > _MODEL_VIEW, include_args=[], include_result=False)
    def change_selected_model(self, message: ListView.Selected | None = None) -> None:
        """Change the currently selected Model."""

    @log_call(action_type="remake_list" > _MODEL_VIEW, include_result=False)
    def remake_model_list(self) -> None:
        """Reconstruct the model list."""

    @log_call(action_type="unselect" > _MODEL_VIEW, include_result=False)
    def no_model_selected(self) -> None:
        """Hide the relavant widgets when no model is selected."""
