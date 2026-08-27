"""The screen for managing of groups."""

from typing import ClassVar, Type

from eliot import log_message
from textual import on
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import HorizontalGroup, VerticalGroup, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Markdown,
    Rule,
    Static,
)

from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from ml_on_apx.logging import CallbackDecorator, log_call, log_with_callback
from ml_on_apx.model_management import _TUI
from ml_on_apx.model_management.app_views.model_view import ModelView
from ml_on_apx.model_management.group_info import GroupInfo
from ml_on_apx.model_management.model_manager import ModelManager
from ml_on_apx.model_management.models.simple_model.simple_model import SimpleGroupInfo
from ml_on_apx.tui_common.binary_modal_question import BinaryModalQuestion
from ml_on_apx.tui_common.get_string_question import GetStringQuestion
from ml_on_apx.tui_common.list_select_question import ListSelectQuestion

"""
-----------------------------------------------------------------------------
|  < group name >   |   < GROUP NAME >                                      |
|  < group name >   |   [ Rename ][ Delete ]                                |
|  < group name >   |   < Group Info ...                                    |
|  < group name >   |                                                       |
|  < group name >   |                    >                                  |
|  < group name >   |   Models:                                             |
|  < group name >   |   - < model name >                                    |
|  < group name >   |   - < model name >                                    |
|  < group name >   |   - < model name >                                    |
|  < group name >   |   - < model name >                                    |
|  [ New Group  ]   |                                                       |
-----------------------------------------------------------------------------
"""

_GROUP_VIEW = "group" @ _TUI

DEFAULT_MESSAGE = """Select a group from the list to the left, or press the
button to create a new one.

Press (control + p) to open the command palette.
"""


class GroupView(Screen[None]):
    """The view for the available groups."""

    BINDINGS: ClassVar[list[BindingType]] = [
        ("N", "new_group", "Create a new group"),
        ("M", "manage_group", "Manage selected group"),
        ("R", "rename_group", "Rename group"),
        ("D", "delete_group", "Delete group"),
        ("P", "new_with_preset", "Clone & Edit"),
    ]

    selected_group: reactive[str | None] = reactive(None, bindings=True)

    def __init__(
        self,
        model_manager: ModelManager,
        dataset_manager: DatasetManager,
        features: list[str],
    ) -> None:
        """Create a new group view.

        Args:
            model_manager (ModelManager): The manager that manages the groups and
                models.
            dataset_manager (DatasetManager): The manager that manages the dataset.
            features (list[str]): The list of available features.

        """
        super(GroupView, self).__init__()
        self._model_manager = model_manager
        self._dataset_manager = dataset_manager
        self._features = features

    def compose(self) -> ComposeResult:
        """Build the screen from its component widgets.

        Returns:
            ComposeResult: An iterator of widgets this screen incorporates.

        Yields:
            Iterator[ComposeResult]: A widget to incorporated.

        """
        yield Header()
        yield Footer()
        with VerticalGroup(id="navigation-panel"):
            yield ListView(id="group-list")
            yield Button("New Group", id="new-group-button")
        with VerticalScroll(classes="container", id="group-info-view"):
            yield Label("", classes="title", id="group-name")
            with HorizontalGroup(id="control-buttons"):
                yield Button("Manage", id="manage-group-button")
                yield Button("Rename", id="rename-group-button")
                yield Button("Clone & Edit", id="preset-group-button")
                yield Button("Delete", variant="error", id="delete-group-button")
            yield Markdown(id="group-info-box")
            with VerticalGroup(id="model-box"):
                yield Rule()
                yield Label("Models:", classes="title", id="models-label")
                yield VerticalGroup(id="model-list")

    async def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""
        await self.remake_group_list()
        self.no_selection_view()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check to see if an action can be performed.

        Args:
            action (str): The action to be performed
            parameters (tuple[object, ...]): The parameters for that action.

        Returns:
            bool | None: If the action can be performed, explicitly cannot be, or cannot
                be and should not be shown (True, False, and None, respectively).

        """
        if action == "new_group":
            return True
        if action in {
            "manage_group",
            "rename_group",
            "delete_group",
            "new_with_preset",
        }:
            return self.selected_group is not None
        return False

    def validate_selected_group(self, new_name: str | None) -> str | None:
        """Validate new values for the reactive component `selected_group`.

        Args:
            new_name (str | None): The new value to be set.

        Returns:
            str | None: The actual value that should be set.

        """
        valid = self._model_manager.group_names
        if new_name not in valid:
            new_name = None
        return new_name

    def watch_selected_group(self, new_name: str | None) -> None:
        """Handle changes in the reactive component `selected_group`.

        Args:
            new_name (str | None): The new value for the component

        """
        if new_name is None:
            self.no_selection_view()
        else:
            control_buttons = self.get_widget_by_id("control-buttons")
            model_info = self.get_widget_by_id("model-box")
            control_buttons.display = True
            model_info.display = True

            title_label = self.get_widget_by_id("group-name", Label)

            content_markdown = self.get_widget_by_id("group-info-box", Markdown)
            group_info = self._model_manager.get_group_info(new_name)

            model_list = self.get_widget_by_id("model-list", VerticalGroup)
            for child in model_list.children:
                child.remove()
            for model in self._model_manager.get_model_names(new_name):
                model_list.mount(Label(f"- {model}"))

            title_label.content = new_name
            content_markdown.update(group_info.get_markdown(self._model_manager))

    @on(Button.Pressed)
    @log_call(
        action_type="button_pressed" > _GROUP_VIEW,
        include_args=[],
        include_result=False,
    )
    async def handle_button_press(self, message: Button.Pressed) -> None:
        """Handle the Pressed event from any child button.

        Args:
            message (Button.Pressed): The event to handle.

        """
        button_id = message.button.id
        match button_id:
            case "new-group-button":
                self.action_new_group()
            case "manage-group-button":
                self.action_manage_group()
            case "rename-group-button":
                self.action_rename_group()
            case "delete-group-button":
                self.action_delete_group()

    @on(Button.Pressed, "#preset-group-button")
    @log_with_callback("new_with_preset" > _GROUP_VIEW)
    def action_new_with_preset(
        self, callback: CallbackDecorator, message: Button.Pressed | None = None
    ) -> None:
        """Create a new group using the current one as a preset."""

        @callback
        async def callback_new_group_preset(result: GroupInfo | None) -> None:
            if result is not None:
                self.write_group(result)
            else:
                log_message("nothing_created")

        if self.selected_group is not None:
            self.app.push_screen(
                self._model_manager.get_group_info(
                    self.selected_group
                ).screen_with_presets(),
                callback=callback_new_group_preset,
            )

    @on(ListView.Selected)
    @log_call(
        action_type="select_grp" > _GROUP_VIEW, include_args=[], include_result=False
    )
    def handle_list_view_selected(self, message: ListView.Selected) -> None:
        """Handle the Selected event from the group list.

        Args:
            message (ListView.Selected): The event to handle.

        """
        if message.item.name is not None:
            self.selected_group = message.item.name
            message.stop()
        else:
            log_message("selected_nothing")

    @log_call(action_type="remake_grp_list" > _GROUP_VIEW, include_result=False)
    async def remake_group_list(self) -> None:
        """Remake and display the list of groups shown to the user."""
        group_list = self.get_widget_by_id("group-list", ListView)
        await group_list.clear()
        group_names = list(self._model_manager.group_names)
        group_names.sort()
        for group_name in group_names:
            group_list.append(ListItem(Label(group_name), name=group_name))

    @log_call(action_type="no_grp_selected" > _GROUP_VIEW, include_result=False)
    def no_selection_view(self) -> None:
        """Set up the screen when no group is selected."""
        title = self.get_widget_by_id("group-name", Label)
        button_group = self.get_widget_by_id("control-buttons")
        markdown = self.get_widget_by_id("group-info-box", Markdown)
        models = self.get_widget_by_id("model-box")

        button_group.display = False
        models.display = False

        title.content = "Group Management"
        markdown.update(DEFAULT_MESSAGE)

    @log_with_callback(action_type="new" > _GROUP_VIEW)
    def action_new_group(self, callback: CallbackDecorator) -> None:
        """Create a new group."""

        @callback
        async def callback_new_group_type(result: Type[GroupInfo] | None) -> None:
            if result is not None:
                self.app.push_screen(
                    result.screen(self._features),
                    callback=lambda x: self.write_group(x) if x is not None else x,
                )

        self.app.push_screen(
            ListSelectQuestion(
                [("Simple (Linear, Sequential) Model", SimpleGroupInfo)]
            ),
            callback=callback_new_group_type,
        )

    @log_with_callback(action_type="rename" > _GROUP_VIEW)
    def action_rename_group(self, callback: CallbackDecorator) -> None:
        """Rename a group."""

        @callback
        async def callback_rename_group(name: str | None) -> None:
            if name:
                if self.selected_group is not None:
                    self._model_manager.rename_group(self.selected_group, name)
                    self.selected_group = name
                    await self.remake_group_list()
                else:
                    log_message("rename_nothing")

        self.app.push_screen(
            GetStringQuestion(
                title=f"Rename {self.selected_group} to ..?",
                validator=GroupInfo.validate_group_name,
            ),
            callback=callback_rename_group,
        )

    @log_with_callback(action_type="del" > _GROUP_VIEW)
    def action_delete_group(self, callback: CallbackDecorator) -> None:
        """Delete a group and all of its models."""

        @callback
        async def callback_delete_group(delete: bool | None) -> None:
            if delete:
                if self.selected_group is not None:
                    self._model_manager.delete_group(self.selected_group)
                    self.selected_group = None
                    await self.remake_group_list()
                else:
                    log_message("delete_nothing")

        self.app.push_screen(
            BinaryModalQuestion(
                Static(
                    "If you do so, all associated models will also be deleted "
                    "and will not be recoverable!",
                ),
                title="Really delete this group?",
            ),
            callback=callback_delete_group,
        )

    @log_with_callback(action_type="manage" > _GROUP_VIEW)
    def action_manage_group(self, callback: CallbackDecorator) -> None:
        """Manage the models within a group."""

        @callback
        async def close(_none: None) -> None:
            pass

        if self.selected_group is not None:
            self.app.push_screen(
                ModelView(
                    self.selected_group, self._model_manager, self._dataset_manager
                ),
                callback=close,
            )

    @log_with_callback(action_type="write_new" > _GROUP_VIEW)
    def write_group(self, callback: CallbackDecorator, group: GroupInfo) -> None:
        """Save a group with the manager."""

        @callback
        async def callback_write_group(name: str | None) -> None:
            if name:
                self._model_manager.create_group(name, group)
                await self.remake_group_list()

        self.app.push_screen(
            GetStringQuestion(
                title="The group name?",
                subtitle="Warning: Canceling will delete any changes!",
                validator=GroupInfo.validate_group_name,
            ),
            callback=callback_write_group,
        )
