from textual.reactive import reactive
from ml_on_apx.dataset_management.app_views import get_dataset_info_markdown
from textual import on
from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from textual.widgets import Header, Footer, ListView, Label, Button, Markdown, ListItem
from textual.containers import VerticalScroll, HorizontalGroup, VerticalGroup
from textual.screen import Screen
from textual.app import ComposeResult


class MainView(Screen[None]):
    BINDINGS = [("")]

    _dataset_name: reactive[str | None] = reactive(None, bindings=True)

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
            with HorizontalGroup(classes="control_buttons"):
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

    def on_mount(self):

        dataset_list = self.get_widget_by_id("dataset-list")
        assert type(dataset_list) is ListView

        for dataset_name in self._manager.get_dataset_names():
            dataset_list.append(ListItem(Label(dataset_name), name=dataset_name))

    @on(Button.Pressed)
    def handle_button_press(self, message: Button.Pressed):
        button_id = message.button.id
        match button_id:
            case "new-dataset-button":
                # push new-edit-dataset view
                pass
            case "edit-dataset-button":
                # push new-edit-dataset view
                pass
            case "rename-dataset-button":
                # push rename screen
                pass
            case "delete-dataset-button":
                # push delete screen
                pass
            case "force-recompile-dataset-button":
                if self._dataset_name is None:
                    raise ValueError("Button somehow pressed with no selected dataset!")
                info = self._manager.get_dataset_info(self._dataset_name)
                self._manager.update_dataset(self._dataset_name, info)

    @on(ListView.Selected)
    def handle_list_view_selected(self, message: ListView.Selected):
        dataset_info_view = self.get_child_by_id("dataset-info-view")
        dataset_info_view.display = True

        title_label = self.get_child_by_id("dataset-name")
        assert type(title_label) is Label

        content_markdown = self.get_child_by_id("dataset-info-box")
        assert type(content_markdown) is Markdown

        assert message.item.name is not None

        self._dataset_name = message.item.name
        dataset_info = self._manager.get_dataset_info(self._dataset_name)

        title_label.content = self._dataset_name
        content_markdown.update(get_dataset_info_markdown(dataset_info, self._manager))

        message.stop()

    def action_new_dataset(self):
        pass

    def action_edit_dataset(self):
        pass

    def action_rename_dataset(self):
        pass

    def action_delete_dataset(self):
        pass

    def action_recompile_dataset(self):
        pass
