from ml_on_apx.dataset_management.app_views.binary_modal_question import (
    BinaryModalQuestion,
)
from textual.reactive import reactive
from ml_on_apx.dataset_management.app_views import get_dataset_info_markdown
from textual import on
from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from textual.widgets import Header, Footer, ListView, Label, Button, Markdown, ListItem
from textual.containers import VerticalScroll, HorizontalGroup, VerticalGroup
from textual.screen import Screen
from textual.app import ComposeResult


DEFAULT_MESSAGE = """Select a dataset from the list to the right, or press the
button to create a new one.

Press (control + p) to view additional information.
"""


class MainView(Screen[None]):
    BINDINGS = [
        ("N", "new_dataset", "Create a new dataset"),
        ("E", "edit_dataset", "Edit dataset"),
        ("R", "rename_dataset", "Rename dataset"),
        ("D", "delete_dataset", "Delete dataset"),
        ("F", "recompile_dataset", "Force the dataset to recompile."),
    ]

    dataset_name: reactive[str | None] = reactive(None, bindings=True)

    def __init__(self, manager: DatasetManager[Dataset]):
        super().__init__()
        self._manager = manager

    def compose(self) -> ComposeResult:
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

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
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
        valid = self._manager.get_dataset_names()
        if new_name not in valid:
            new_name = None
        return new_name

    def watch_dataset_name(self, new_name: str | None) -> str | None:
        if new_name is None:
            self.no_selection_view()
        else:
            dataset_info_view = self.get_child_by_id("dataset-info-view")
            dataset_info_view.display = True

            title_label = self.get_child_by_id("dataset-name")
            assert type(title_label) is Label

            content_markdown = self.get_child_by_id("dataset-info-box")
            assert type(content_markdown) is Markdown
            dataset_info = self._manager.get_dataset_info(new_name)

            title_label.content = new_name
            content_markdown.update(
                get_dataset_info_markdown(dataset_info, self._manager)
            )

    def on_mount(self):
        self.remake_dataset_list()
        self.no_selection_view()

    @on(Button.Pressed)
    def handle_button_press(self, message: Button.Pressed):
        button_id = message.button.id
        match button_id:
            case "new-dataset-button":
                self.action_new_dataset()
                pass
            case "edit-dataset-button":
                self.action_edit_dataset()
                pass
            case "rename-dataset-button":
                self.action_rename_dataset()
                pass
            case "delete-dataset-button":
                self.action_delete_dataset()
                pass
            case "force-recompile-dataset-button":
                self.action_recompile_dataset()

    @on(ListView.Selected)
    def handle_list_view_selected(self, message: ListView.Selected):
        assert message.item.name is not None
        self.dataset_name = message.item.name
        message.stop()

    def action_new_dataset(self):
        pass

    def action_edit_dataset(self):
        pass

    def action_rename_dataset(self):
        pass

    def action_delete_dataset(self):
        def check_delete(delete: bool | None):
            if delete:
                assert self.dataset_name is not None
                self._manager.delete_dataset(self.dataset_name)
                self.dataset_name = None

        self.app.push_screen(
            BinaryModalQuestion(
                Label(
                    f"Are you sure you want to delete the dataset '{self.dataset_name}'?"
                )
            ),
            check_delete,
        )

    def action_recompile_dataset(self):
        if self.dataset_name is None:
            raise ValueError("Button somehow pressed with no selected dataset!")
        info = self._manager.get_dataset_info(self.dataset_name)
        self._manager.update_dataset(self.dataset_name, info)

    def remake_dataset_list(self):
        dataset_list = self.get_widget_by_id("dataset-list")
        assert type(dataset_list) is ListView

        for dataset_name in self._manager.get_dataset_names():
            dataset_list.append(ListItem(Label(dataset_name), name=dataset_name))

    def no_selection_view(self):
        title = self.get_widget_by_id("dataset-name")
        button_group = self.get_widget_by_id("control-buttons")
        markdown = self.get_widget_by_id("dataset-info-box")
        recomp_button = self.get_widget_by_id("force-recompile-dataset-button")

        button_group.display = False
        recomp_button.display = False

        assert type(title) is Label
        assert type(markdown) is Markdown

        title.content = "Dataset Management"
        markdown.update(DEFAULT_MESSAGE)
