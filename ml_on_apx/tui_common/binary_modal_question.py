"""A question with a binary answer."""

from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalGroup, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button

from ml_on_apx.logging import log_call
from ml_on_apx.tui_common import _TUI

_BMQ = "bmp" @ _TUI


class BinaryModalQuestion(ModalScreen[bool]):
    """A question with a binary answer."""

    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("escape", "exit", "Exit")
    ]

    def __init__(
        self,
        *questions: Widget,
        title: str | None = None,
        subtitle: str | None = None,
        true_button_label: Content | Text | str | None = "Yes",
        false_button_label: Content | Text | str | None = "No",
    ) -> None:
        """Create a new BinaryModelQuestion.

        Args:
            *questions (Widget): A set of widget to put in the question box.
            title (str | None, optional): The title of the box. Defaults to None.
            subtitle (str | None, optional): The subtitle around the box. Defaults to
                None.
            true_button_label (Content | Text | str | None, optional): The label for the
                button that returns true. Defaults to "Yes".
            false_button_label (Content | Text | str | None, optional): The label for
                the button that returns false. Defaults to "No".

        """
        super().__init__()
        self._questions = questions
        self._title = title
        self._subtitle = subtitle
        self._true_button_label = true_button_label
        self._false_button_label = false_button_label

    def compose(self) -> ComposeResult:
        """Build the screen from its component widgets.

        Returns:
            ComposeResult: An iterator of widgets this screen incorporates.

        Yields:
            Iterator[ComposeResult]: A widget to incorporated.

        """
        with VerticalGroup(classes="container", id="container"):
            yield VerticalScroll(*self._questions, classes="question")
            yield Button(self._true_button_label, variant="success", id="bmodal-true")
            yield Button(self._false_button_label, variant="error", id="bmodal-false")

    @log_call(action_type="mount" @ _BMQ)
    def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""
        container = self.get_child_by_id("container")
        container.border_title = self._title
        container.border_subtitle = self._subtitle
        container.get_child_by_id("bmodal-true").focus()

    @log_call(action_type="exit" @ _BMQ)
    def action_exit(self) -> None:
        """Process the action `exit`."""
        self.dismiss(None)

    @on(Button.Pressed)
    @log_call(action_type="button" @ _BMQ)
    def handle_button_press(self, message: Button.Pressed) -> None:
        """Handle the Pressed event from a child button.

        Args:
            message (Button.Pressed): The event to handle.

        """
        button_id = message.button.id
        match button_id:
            case "bmodal-true":
                self.dismiss(True)
            case "bmodal-false":
                self.dismiss(False)
