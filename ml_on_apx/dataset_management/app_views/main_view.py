from textual import on
from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from textual.widgets import Header, Footer, ListView, Label, Button, Markdown, ListItem
from textual.containers import VerticalScroll, HorizontalGroup, VerticalGroup
from textual.screen import Screen
from textual.app import ComposeResult


class MainView(Screen[None]):
    def __init__(self, manager: DatasetManager[Dataset]):
        super().__init__()
        self._manager = manager

    def compose(self) -> ComposeResult:
        self._dataset_list = ListView(id="dataset-list")

        yield Header(name="Manage Datasets")
        with VerticalGroup(id="navigation-panel"):
            yield self._dataset_list
            yield Button("New Dataset")
        with VerticalScroll(classes="container", id="dataset-info-view"):
            yield Label("", classes="title", id="dataset-name")
            with HorizontalGroup(classes="control_buttons"):
                yield Button("Edit")
                yield Button("Rename")
                yield Button("Delete", variant="error")
            yield Markdown(id="dataset-info-box")
            yield Button("Force Recompile", variant="warning", classes="bottom")
        yield Footer()

    def on_mount(self):
        dataset_info_view = self.get_child_by_id("dataset-info-view")
        dataset_info_view.display = False

        for dataset_name in self._manager.get_dataset_names():
            self._dataset_list.append(ListItem(Label(dataset_name), name=dataset_name))

    @on(ListView.Selected)
    def handle_list_view_selected(self, message: ListView.Selected):
        dataset_info_view = self.get_child_by_id("dataset-info-view")
        dataset_info_view.display = True

        title_label = self.get_child_by_id("dataset-name")
        if type(title_label) is not Label:
            raise ValueError("Item with id `#dataset-name` must be a `Label`!")

        content_markdown = self.get_child_by_id("dataset-info-box")
        if type(content_markdown) is not Markdown:
            raise ValueError("Item with id `#dataset-info-box` must be a `Markdown`!")

        if message.item.name is None:
            raise ValueError("Dataset info list item must have its name set!")

        name = message.item.name
        info = self._manager.get_dataset_info(name)
        labels = info.get_labels()
        sources = info.get_labeled_sources()

        markdown = "The dataset uses the following labels:\n"
        for label in labels:
            markdown += f"  - {label}\n"
        markdown += "\nThe dataset uses the following sources:\n"
        for source in sources:
            markdown += f"  - {source[0].relative_to(self._manager.get_root_dir_path())} ({source[1]})\n"

        title_label.content = name
        content_markdown.update(markdown)

        message.stop()
