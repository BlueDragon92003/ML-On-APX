"""The TUI app the user uses to manage datasets, plus the way to run it."""

from pathlib import Path
from typing import ClassVar, Type

import textual.widgets
from eliot import start_action
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Label

from ml_on_apx.dataset_management.app_views.main_view import MainView
from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from ml_on_apx.logging import log_call
from ml_on_apx.modes import Mode
from ml_on_apx.tui_common.binary_modal_question import (
    BinaryModalQuestion,
)


class DatasetManagerApp(App):
    """The TUI app used to manage datasets."""

    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("q", "show_quit_screen", "Quit")
    ]
    CSS_PATH = "./app_views/app.tcss"

    def __init__(self, active_dataset_manager: DatasetManager) -> None:
        """Create a new Dataset Manager App.

        Args:
            active_dataset_manager (DatasetManager): The dataset manager this app will
                use to manage datasets.

        """
        super().__init__()
        self._manager = active_dataset_manager

    def compose(self) -> ComposeResult:
        """Build the screen from its component widgets.

        Returns:
            ComposeResult: An iterator of widgets this screen incorporates.

        Yields:
            Iterator[ComposeResult]: A widget to incorporated.

        """
        yield textual.widgets.LoadingIndicator()

    @log_call(action_type="data:app:app:mount")
    async def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""
        self.theme = "gruvbox"
        self.push_screen(MainView(self._manager))

    @log_call(action_type="data:app:app:quit_screen")
    def action_show_quit_screen(self) -> None:
        """Process the action `show_quit_screen`."""
        action = start_action(action_type="data:app:app:show_quit")

        def check_quit(sentinal: bool | None) -> None:
            with action.context():
                if sentinal:
                    self.exit()
            action.finish()

        with action.context():
            self.push_screen(
                BinaryModalQuestion(Label("Quit dataset management?")), check_quit
            )


@log_call(action_type="data:app:start")
def main(
    dataset_dir: Path,
    mode: Mode,
    dataset_class: Type[Dataset],
) -> None:
    """Run the dataset manager app.

    Args:
        dataset_dir (Path): The directory all dataset information is stored under.
        mode (Mode): The mode of datasets this app is managing.
        dataset_class (Type[Dataset]): The class that should be used for datasets.
        log_level (LogLevel): The logging level to use.

    """
    with DatasetManager(dataset_dir, mode, dataset_class) as manager:
        app = DatasetManagerApp(manager)
        app.run()
