"""The screen for managing models within a group."""

from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import HorizontalGroup, VerticalGroup, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    ListItem,
    ListView,
    Markdown,
    Rule,
)
from textual.widgets import Label as TuiLabel

from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from ml_on_apx.logging import CallbackDecorator, log_call, log_with_callback
from ml_on_apx.model_management import _TUI
from ml_on_apx.model_management.app_views.training_job_screen import (
    CreateTrainingJobScreen,
)
from ml_on_apx.model_management.model_manager import ModelManager
from ml_on_apx.model_management.training_job import TrainingJob
from ml_on_apx.tui_common.binary_modal_question import BinaryModalQuestion
from ml_on_apx.tui_common.get_string_question import GetStringQuestion

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

DEFAULT_MESSAGE = """Select a model from the list to the left, or press the
button to create a new one.

Press (control + p) to open the command palette.
"""


class ModelView(Screen[None]):
    """A view for the available models in a group."""

    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "back", "Back"),
        ("N", "train_model", "Train new model"),
        ("R", "rename_model", "Rename model"),
        ("D", "delete_model", "Delete model"),
        ("T", "test_model", "Test model"),
    ]

    selected_model: reactive[str | None] = reactive(None)

    def __init__(
        self,
        group_name: str,
        model_manager: ModelManager,
        dataset_manager: DatasetManager,
    ) -> None:
        """Create a new ModelView."""
        super(ModelView, self).__init__()
        self._group_name = group_name
        self._model_manager = model_manager
        self._dataset_manager = dataset_manager

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
            with VerticalGroup(id="test-box"):
                yield Rule()
                yield TuiLabel("Test Results", classes="title", id="tests-title")
                yield Button("Run test", variant="primary", id="run-test-button")
                yield VerticalGroup(id="test-list")

    async def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""
        await self.remake_model_list()
        self.no_model_selected()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check to see if an action can be performed."""
        match action:
            case "back" | "train_model":
                return True
            case "rename_model" | "delete_model" | "test_model":
                return self.selected_model is not None
        return False

    def validate_selected_model(self, new_name: str | None) -> str | None:
        """Validate new values for the reactive component `selected_model`."""
        valid = self._model_manager.get_model_names(self._group_name)
        if new_name not in valid:
            new_name = None
        return new_name

    def watch_selected_model(self, new_val: str | None) -> None:
        """Handle changes in the reactive component `selected_model`."""
        if new_val is None:
            self.no_model_selected()
        else:
            model_info = self._model_manager.get_model_info(self._group_name, new_val)

            self.get_widget_by_id("control-buttons").display = True
            self.get_widget_by_id("test-box").display = True

            self.get_widget_by_id("model-name", TuiLabel).content = new_val
            self.get_widget_by_id("model-info-box", Markdown).update(
                model_info.markdown
            )

            tests = self.get_widget_by_id("test-list", VerticalGroup)
            for test in model_info.testing_information:
                tests.mount(
                    Collapsible(Markdown(test.markdown), title=test.test_time.ctime())
                )

    @on(Button.Pressed, "#return-button")
    @log_call(action_type="back" > _MODEL_VIEW, include_args=[], include_result=False)
    def action_back(self, message: Button.Pressed | None = None) -> None:
        """Return to the group view."""
        self.dismiss()

    @on(Button.Pressed, "#new-model-button")
    @log_with_callback(action_type="train" > _MODEL_VIEW, include_caller_args=[])
    def action_train_model(
        self, callback: CallbackDecorator, message: Button.Pressed | None = None
    ) -> None:
        """Set the TrainingJob."""

        @callback
        async def callback_train(job: TrainingJob | None) -> None:
            if job is None:
                self.app.notify("Job discarded.")
            else:
                self._model_manager.training_job = job
                self.app.notify("Set trianing job.")

        self.app.push_screen(
            CreateTrainingJobScreen(
                list(self._dataset_manager.dataset_names),
                self._model_manager.get_model_names(self._group_name),
                self._group_name,
                self._model_manager.get_group_info(self._group_name).get_labels(
                    self._model_manager
                ),
                {
                    x: self._dataset_manager.get_dataset_info(x).labels
                    for x in self._dataset_manager.dataset_names
                },
            ),
            callback=callback_train,
        )

    @on(Button.Pressed, "#rename-model-button")
    @log_with_callback(
        action_type="rename" > _MODEL_VIEW,
        include_caller_args=[],
    )
    def action_rename_model(
        self, callback: CallbackDecorator, message: Button.Pressed | None = None
    ) -> None:
        """Rename the current Model."""

        @callback
        async def callback_rename(new_name: str | None) -> None:
            if new_name and self.selected_model is not None:
                # TODO static model name validation
                self._model_manager.rename_model(
                    self._group_name, self.selected_model, new_name
                )
                await self.remake_model_list()
                self.selected_model = new_name

        self.app.push_screen(
            GetStringQuestion(
                title=f"Rename {self.selected_model} to?"
                # validator=
            ),
            callback=callback_rename,
        )

    @on(Button.Pressed, "#delete-model-button")
    @log_with_callback(action_type="delete" > _MODEL_VIEW, include_caller_args=[])
    def action_delete_model(
        self, callback: CallbackDecorator, message: Button.Pressed | None = None
    ) -> None:
        """Delete the current Model."""

        @callback
        async def callback_delete_model(sentinal: bool | None) -> None:
            if sentinal and self.selected_model is not None:
                self._model_manager.delete_model(self._group_name, self.selected_model)
                await self.remake_model_list()
                self.selected_model = None

        self.app.push_screen(
            BinaryModalQuestion(TuiLabel(f"Really delete {self.selected_model}?")),
            callback=callback_delete_model,
        )

    @on(Button.Pressed, "#run-test-button")
    @log_call(action_type="test" > _MODEL_VIEW, include_args=[], include_result=False)
    def action_test_model(self, message: Button.Pressed | None = None) -> None:
        """Set the TestingJob."""
        # TODO testing_job_screen

    @on(ListView.Selected, "#model-list")
    @log_call(action_type="select" > _MODEL_VIEW, include_args=[], include_result=False)
    def change_selected_model(self, message: ListView.Selected) -> None:
        """Change the currently selected Model."""
        if message.item.name is not None:
            self.selected_model = message.item.name

    @log_call(action_type="remake_list" > _MODEL_VIEW, include_result=False)
    async def remake_model_list(self) -> None:
        """Reconstruct the model list."""
        model_list = self.get_widget_by_id("model-list", ListView)
        await model_list.clear()
        group_names = list(self._model_manager.get_model_names(self._group_name))
        group_names.sort()
        for group_name in group_names:
            model_list.append(ListItem(TuiLabel(group_name), name=group_name))

    @log_call(action_type="unselect" > _MODEL_VIEW, include_result=False)
    def no_model_selected(self) -> None:
        """Hide the relavant widgets when no model is selected."""
        self.get_widget_by_id("control-buttons").display = False
        self.get_widget_by_id("test-box").display = False

        self.get_widget_by_id("model-name", TuiLabel).content = "Model Management"
        self.get_widget_by_id("model-info-box", Markdown).update(DEFAULT_MESSAGE)

        for child in self.get_widget_by_id("test-list", VerticalGroup).children:
            child.remove()
