from textual.widgets import Label
from ml_on_apx.dataset_management.app_views.binary_modal_question import (
    BinaryModalQuestion,
)
import textual.widgets
from ml_on_apx.dataset_management.app_views.main_view import MainView
from typing import Type
from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.modes import Mode
from pathlib import Path
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from textual.app import App, ComposeResult


class DatasetManagerApp(App):
    BINDINGS = [("q", "show_quit_screen", "Quit")]
    CSS_PATH = "./app_views/app.tcss"

    def __init__(self, active_dataset_manager: DatasetManager):
        super().__init__()
        self._manager = active_dataset_manager

    def compose(self) -> ComposeResult:
        yield textual.widgets.LoadingIndicator()

    async def on_mount(self):
        self.theme = "gruvbox"
        self.push_screen(MainView(self._manager))

    def action_show_quit_screen(self):
        def check_quit(sentinal: bool | None):
            if sentinal:
                self.exit()

        self.push_screen(
            BinaryModalQuestion(Label("Quit dataset management?")), check_quit
        )


def main(dataset_dir: Path, mode: Mode, dataset_class: Type[Dataset]):
    with DatasetManager(dataset_dir, mode, dataset_class) as manager:
        app = DatasetManagerApp(manager)
        app.run()
