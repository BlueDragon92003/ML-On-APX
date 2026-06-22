import textual.widgets
from ml_on_apx.dataset_management.app_views.main_view import MainView
from typing import Type
from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.modes import Mode
from pathlib import Path
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from textual.app import App, ComposeResult


class DatasetManagerApp(App):
    BINDINGS = [("q", "quit", "Quit")]
    CSS_PATH = "./app_views/app.tcss"

    def __init__(self, active_dataset_manager: DatasetManager):
        super().__init__()
        self._manager = active_dataset_manager

    def compose(self) -> ComposeResult:
        yield textual.widgets.LoadingIndicator()

    def on_mount(self):
        self.theme = "gruvbox"
        self.push_screen(MainView(self._manager))


def main(dataset_dir: Path, mode: Mode, dataset_class: Type[Dataset]):
    with DatasetManager(dataset_dir, mode, dataset_class) as manager:
        app = DatasetManagerApp(manager)
        app.run()
        print("(Re)compiling dataset pickles, if any need to be...")
