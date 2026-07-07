"""A question where the user provides a string."""

from typing import Callable, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalGroup
from textual.screen import ModalScreen
from textual.widgets import Input

from ml_on_apx.logging import log_call
from ml_on_apx.tui_common import _TUI

_GSQ = "gsq" @ _TUI


class GetStringQuestion(ModalScreen[str]):
    """A question with a text answer."""

    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("escape", "exit", "Cancel")
    ]

    def __init__(
        self,
        validator: Callable[[str], bool] = lambda _: True,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> None:
        """Initialize a new GetStringQuestion.

        Args:
            validator (Callable[[str], bool], optional): A validator for the string
                input. Defaults to `lambda _: True`.
            title (str | None, optional): The title of the box. Defaults to None.
            subtitle (str | None, optional): The subtitle around the box. Defaults to
                None.

        """
        super().__init__()
        self._validator = validator
        self._title = title
        self._subtitle = subtitle

    def compose(self) -> ComposeResult:
        """Build the screen from its component widgets.

        Returns:
            ComposeResult: An iterator of widgets this screen incorporates.

        Yields:
            Iterator[ComposeResult]: A widget to incorporated.

        """
        with VerticalGroup(classes="container", id="container"):
            yield Input(id="getstr-input")

    @log_call(action_type="mount" @ _GSQ)
    def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""
        container = self.get_child_by_id("container")
        container.border_title = self._title
        container.border_subtitle = self._subtitle
        container.get_child_by_id("getstr-input").focus()

    @log_call(action_type="exit" @ _GSQ)
    def action_exit(self) -> None:
        """Process the action `exit`."""
        self.dismiss(None)

    @on(Input.Submitted)
    @log_call(action_type="submit" @ _GSQ)
    def handle_input_submission(self, message: Input.Submitted) -> None:
        """Handle the Submitted event from a descendant Input widget.

        Propogates:
            Whatever the validator can raise.

        Args:
            message (Button.Pressed): The event to handle.

        """
        if self._validator(message.value):
            self.dismiss(message.value)
        else:
            self.app.notify("Not a valid input!")
