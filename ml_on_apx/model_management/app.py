"""The TUI app the user uses to manage managers, plus the way to run it."""

from pathlib import Path
from typing import ClassVar

import textual.widgets
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Label

from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from ml_on_apx.logging import CallbackDecorator, log_call, log_with_callback
from ml_on_apx.model_management import _APP, _TUI
from ml_on_apx.model_management.app_views.group_view import GroupView
from ml_on_apx.model_management.model_manager import ModelManager
from ml_on_apx.modes import Mode
from ml_on_apx.tui_common.binary_modal_question import BinaryModalQuestion


class ModelManagerApp(App):
    """The TUI app used to manage models."""

    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("q", "show_quit_screen", "Quit")
    ]
    CSS_PATH = [  # noqa: RUF012
        "./app_views/app.tcss",
        "./models/simple_model/simple.tcss",
        "../tui_common/common.tcss",
    ]

    def __init__(
        self,
        active_model_manager: ModelManager,
        active_dataset_manager: DatasetManager,
        features: list[str],
    ) -> None:
        """Create a new Dataset Manager App.

        Args:
            active_model_manager (ModelManager): The model manager this app will
                use to manage models.
            active_dataset_manager (DatasetManager): The dataset manager this app will
                use to read info from datasets.
            features (list[str]): The possible features available to the models.

        """
        super().__init__()
        self._model_manager = active_model_manager
        self._dataset_manager = active_dataset_manager
        self._features = features

    def compose(self) -> ComposeResult:
        """Build the screen from its component widgets.

        Returns:
            ComposeResult: An iterator of widgets this screen incorporates.

        Yields:
            Iterator[ComposeResult]: A widget to incorporated.

        """
        yield textual.widgets.LoadingIndicator()

    async def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""
        self.theme = "gruvbox"
        self.push_screen(
            GroupView(self._model_manager, self._dataset_manager, self._features)
        )

    @log_with_callback(action_type="quit" > _APP)
    def action_show_quit_screen(self, callback: CallbackDecorator) -> None:
        """Process the action `show_quit_screen`."""

        @callback
        async def check_quit(sentinal: bool | None) -> None:
            if sentinal:
                self.exit()

        self.push_screen(
            BinaryModalQuestion(Label("Quit dataset management?")), callback=check_quit
        )


@log_call(action_type="start" > _TUI, include_result=False)
def main(
    model_dir: Path,
    dataset_dir: Path,
    mode: Mode,
) -> None:
    """Run the dataset manager app.

    Args:
        model_dir (Path): The directory all model information is stored under.
        dataset_dir (Path): The directory all dataset information is stored under.
        mode (Mode): The mode of models this app is managing.

    """
    # datasets: list[DatasetInfo] = []

    # class FalseDataset(Dataset):
    #     pass

    # with DatasetManager(dataset_dir, mode, FalseDataset) as data_manager:
    #     for name in data_manager.dataset_names:
    #         datasets.append(data_manager.get_dataset_info(name))

    with ModelManager(model_dir, mode) as model_manager:
        with DatasetManager(dataset_dir, mode, mode.dataset_class) as data_manager:
            app = ModelManagerApp(model_manager, data_manager, mode.features)
            app.run()
