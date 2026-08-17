"""The main screen for creating a simple model."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header

from ml_on_apx.model_management.models.simple_model import _SIMPLE
from ml_on_apx.model_management.models.simple_model.layer_widgets import (
    HiddenLayerWidget,
    InputLayerWidget,
    OutputLayerWidget,
)
from ml_on_apx.model_management.models.simple_model.simple_model import SimpleGroupInfo

_TUI = "tui" @ _SIMPLE


class SimpleScreen(Screen[SimpleGroupInfo]):
    """A screen that creates a SimpleGroupInfo object."""

    def __init__(
        self, features: list[str], base: SimpleGroupInfo | None = None
    ) -> None:
        """Create a new screen.

        Args:
            labels (Labels): The labels this group uses.
            features (list[str]): The features this group uses.
            base (SimpleGroupInfo | None, optional): A base GroupInfo to build on or
                alter. Defaults to None.

        """
        super(Screen, self).__init__()
        self._features = features
        self._base = base

    def compose(self) -> ComposeResult:
        """Build the screen from its component widgets.

        Returns:
            ComposeResult: An iterator of widgets this screen incorporates.

        Yields:
            Iterator[ComposeResult]: A widget to incorporated.

        """
        yield Header()
        yield Footer()
        with VerticalScroll(id="hidden-layers"):
            yield InputLayerWidget(self._features, id="input")
            yield OutputLayerWidget(id="output")
        yield Button("Cancel.")
        yield Button("Save...", variant="primary")

    def on_mount(self) -> None:
        """Finish setup of the widget once it is attached to the DOM."""
        if self._base is not None:
            box = self.get_child_by_id("hidden-layers", expect_type=VerticalScroll)
            input_layer = box.get_child_by_type(InputLayerWidget)
            output_layer = box.get_child_by_type(OutputLayerWidget)
            input_layer.all_features = self._base.all_features
            input_layer.features = self._base.features
            output_layer.labels = list(self._base.labels)
            for hl in range(self._base.layer_count):
                if hl == 0 or hl == self._base.layer_count - 1:
                    continue
                hidden_layer = HiddenLayerWidget(id=f"hl{hl}")
                hidden_layer.activation = self._base.get_layer_activation(hl)
                hidden_layer.size = self._base.get_layer_size(hl)
                box.mount(hidden_layer, before="OutputLayerWidget")
