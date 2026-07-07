"""The main view for the dataset management TUI app and associated information."""

import re
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import HorizontalGroup, VerticalGroup, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, ListItem, ListView, Markdown

from ml_on_apx.dataset_management.app_views import get_dataset_info_markdown
from ml_on_apx.dataset_management.app_views.new_edit_view import NewEditView
from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.dataset_management.dataset_info import DATASET_NAME_REGEX
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from ml_on_apx.logging import log_call
from ml_on_apx.tui_common.binary_modal_question import (
    BinaryModalQuestion,
)
from ml_on_apx.tui_common.get_string_question import GetStringQuestion

DEFAULT_MESSAGE = """Select a dataset from the list to the right, or press the
button to create a new one.

Press (control + p) to view additional information.
"""


class MainView(Screen[None]):
    """The main view for the dataset management TUI."""

    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("N", "new_dataset", "Create a new dataset"),
        ("E", "edit_dataset", "Edit dataset"),
        ("R", "rename_dataset", "Rename dataset"),
        ("D", "delete_dataset", "Delete dataset"),
        ("F", "recompile_dataset", "Force the dataset to recompile."),
    ]

    dataset_name: reactive[str | None] = reactive(None, bindings=True)

    def __init__(self, manager: DatasetManager[Dataset]) -> None:
        """Initialize the main view.

        Args:
            manager (DatasetManager[Dataset]): The DatasetManager the app uses to manage
                the datasets.

        """
        super().__init__()
        self._manager = manager

    def compose(self) -> ComposeResult:
        """Build the screen from its component widgets.

        Returns:
            ComposeResult: An iterator of widgets this screen incorporates.

        Yields:
            Iterator[ComposeResult]: A widget to incorporated.

        """
        yield Header(name="Manage Datasets")
        with VerticalGroup(id="navigation-panel"):
            yield ListView(id="dataset-list")
            yield Button("New Dataset", id="new-dataset-button")
        with VerticalScroll(classes="container", id="dataset-info-view"):
            yield Label("", classes="title", id="dataset-name")
            with HorizontalGroup(id="control-buttons"):
                yield Button("Edit", id="edit-dataset-button")
                yield Button("Rename", id="rename-dataset-button")
                yield Button("Delete", variant="error", id="delete-dataset-button")
            yield Markdown(id="dataset-info-box")
            yield Button(
                "Force Recompile",
                variant="warning",
                id="force-recompile-dataset-button",
                classes="bottom",
            )
        yield Footer()

    @log_call(action_type="data:app:main:check_action", include_args=["action"])
    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check to see if an action can be performed.

        Args:
            action (str): The action to be performed
            parameters (tuple[object, ...]): The parameters for that action.

        Returns:
            bool | None: If the action can be performed, explicitly cannot be, or cannot
                be and should not be shown (True, False, and None, respectively).

        """
        if action == "new_dataset":
            return True
        if (
            action == "edit_dataset"
            or action == "rename_dataset"
            or action == "delete_dataset"
            or action == "recompile_dataset"
        ):
            return self.dataset_name is not None

    def validate_dataset_name(self, new_name: str | None) -> str | None:
        """Validate new values for the reactive component `dataset_name`.

        Args:
            new_name (str | None): The new value to be set.

        Returns:
            str | None: The actual value that should be set.

        """
        valid = self._manager.get_dataset_names()
        if new_name not in valid:
            new_name = None
        return new_name

    def watch_dataset_name(self, new_name: str | None) -> None:
        """Handle changes in the reactive component `dataset_name`.

        Args:
            new_name (str | None): The new value for the component

        """
        if new_name is None:
            self.no_selection_view()
        else:
            control_buttons = self.get_widget_by_id("control-buttons")
            recomp_button = self.get_widget_by_id("force-recompile-dataset-button")
            control_buttons.display = True
            recomp_button.display = True

            title_label = self.get_widget_by_id("dataset-name", Label)

            content_markdown = self.get_widget_by_id("dataset-info-box", Markdown)
            dataset_info = self._manager.get_dataset_info(new_name)

            title_label.content = new_name
            content_markdown.update(
                get_dataset_info_markdown(dataset_info, self._manager)
            )

    @log_call(action_type="data:app:main:mount")
    async def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""
        await self.remake_dataset_list()
        self.no_selection_view()

    @on(Button.Pressed)
    @log_call(action_type="data:app:main:button_pressed")
    async def handle_button_press(self, message: Button.Pressed) -> None:
        """Handle the Pressed event from any child button.

        Args:
            message (Button.Pressed): The event to handle.

        """
        button_id = message.button.id
        match button_id:
            case "new-dataset-button":
                self.action_new_dataset()
            case "edit-dataset-button":
                self.action_edit_dataset()
            case "rename-dataset-button":
                self.action_rename_dataset()
            case "delete-dataset-button":
                self.action_delete_dataset()
            case "force-recompile-dataset-button":
                self.action_recompile_dataset()

    @on(ListView.Selected)
    @log_call(action_type="data:app:main:select_ds")
    def handle_list_view_selected(self, message: ListView.Selected) -> None:
        """Handle the Selected event from the dataset list.

        Args:
            message (ListView.Selected): The event to handle.

        """
        assert message.item.name is not None
        self.dataset_name = message.item.name
        message.stop()

    @log_call(action_type="data:app:main:new_ds")
    def action_new_dataset(self) -> None:
        """Process the action `new_dataset`."""

        async def callback(_: None) -> None:
            await self.remake_dataset_list()

        self.app.push_screen(NewEditView(self._manager), callback=callback)

    @log_call(action_type="data:app:main:edit_ds")
    def action_edit_dataset(self) -> None:
        """Process the action `edit_dataset`."""

        async def callback(_: None) -> None:
            await self.remake_dataset_list()

        template_name = self.dataset_name
        assert template_name is not None
        template = self._manager.get_dataset_info(template_name)

        self.app.push_screen(
            NewEditView(self._manager, template=template, template_name=template_name),
            callback=callback,
        )

    @log_call(action_type="data:app:main:rename_ds")
    def action_rename_dataset(self) -> None:
        """Process the action `rename_dataset`."""

        async def callback(new_name: str | None) -> None:
            if new_name == self.dataset_name:
                self.app.notify("The dataset is already named that.")
                return
            if new_name:
                assert self.dataset_name is not None
                try:
                    self._manager.rename_dataset(self.dataset_name, new_name)
                except ValueError:
                    self.app.notify(
                        f"A dataset with the name `{new_name}` already exists",
                        severity="error",
                    )
                else:
                    await self.remake_dataset_list()
                    self.dataset_name = new_name

        self.app.push_screen(
            GetStringQuestion(
                lambda name: re.fullmatch(DATASET_NAME_REGEX, name) is not None,
                title=f"Rename dataset `{self.dataset_name}` to?",
                subtitle="Press escape to cancel.",
            ),
            callback=callback,
        )

    @log_call(action_type="data:app:main:delete_ds")
    def action_delete_dataset(self) -> None:
        """Process the action `delete_dataset`."""

        async def check_delete(delete: bool | None) -> None:
            if delete:
                assert self.dataset_name is not None
                self._manager.delete_dataset(self.dataset_name)
                await self.remake_dataset_list()
                self.dataset_name = None

        self.app.push_screen(
            BinaryModalQuestion(
                Label(f"Are you sure you want to delete `{self.dataset_name}`?")
            ),
            check_delete,
        )

    @log_call(action_type="data:app:main:force_recompile")
    def action_recompile_dataset(self) -> None:
        """Process the action `recompile_dataset`."""
        if self.dataset_name is None:
            return
        info = self._manager.get_dataset_info(self.dataset_name)
        self._manager.update_dataset(self.dataset_name, info)

    @log_call(action_type="data:app:main:remake_ds_list")
    async def remake_dataset_list(self) -> None:
        """Remake and display the list of datasets shown to the user."""
        dataset_list = self.get_widget_by_id("dataset-list", ListView)
        await dataset_list.clear()
        dataset_names = list(self._manager.get_dataset_names())
        dataset_names.sort()
        for dataset_name in dataset_names:
            dataset_list.append(ListItem(Label(dataset_name), name=dataset_name))

    @log_call(action_type="data:app:main:no_ds_selected")
    def no_selection_view(self) -> None:
        """Set up the screen when no dataset is selected."""
        title = self.get_widget_by_id("dataset-name", Label)
        button_group = self.get_widget_by_id("control-buttons")
        markdown = self.get_widget_by_id("dataset-info-box", Markdown)
        recomp_button = self.get_widget_by_id("force-recompile-dataset-button")

        button_group.display = False
        recomp_button.display = False

        title.content = "Dataset Management"
        markdown.update(DEFAULT_MESSAGE)
