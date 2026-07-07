"""A question where the user selects from a list of options."""

from typing import ClassVar, TypeVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalGroup
from textual.screen import ModalScreen
from textual.types import NoSelection
from textual.widgets import Select

from ml_on_apx.logging import log_call
from ml_on_apx.tui_common import _TUI

_LSQ = "lsq" @ _TUI

ListItem = TypeVar("ListItem")


class ListSelectQuestion(ModalScreen[ListItem]):
    """A question with a finite set of answers."""

    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("escape", "exit", "Cancel")
    ]

    def __init__(
        self,
        options: list[tuple[str, ListItem]],
        title: str | None = None,
        subtitle: str | None = None,
    ) -> None:
        """Initialize a new ListSelectQuestion.

        Args:
            options (list[tuple[str, ListItem]]): A list of options the user can select
                from.
            title (str | None, optional): The title of the box. Defaults to None.
            subtitle (str | None, optional): The subtitle around the box. Defaults to
                None.

        """
        super().__init__()
        self._options = options
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
            yield Select[ListItem](self._options, id="slist-list")

    @log_call(action_type="mount" @ _LSQ)
    def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""
        container = self.get_child_by_id("container")
        container.border_title = self._title
        container.border_subtitle = self._subtitle
        container.get_child_by_id("slist-list").focus()

    @log_call(action_type="exit" @ _LSQ)
    def action_exit(self) -> None:
        """Process the action `exit`."""
        self.dismiss(None)

    @on(Select.Changed)
    @log_call(action_type="handle_selection" @ _LSQ)
    def handle_selection(self, message: Select.Changed) -> None:
        """Handle the Changed event from a descendant Select widget.

        Args:
            message (Select.Changed): The event to handle.

        """
        assert type(message) is Select[ListItem].Changed
        value: ListItem | NoSelection = message.value  # ty: ignore[invalid-assignment]
        if type(value) is not NoSelection:
            self.dismiss(value)  # ty: ignore[invalid-argument-type]
        else:
            self.dismiss(None)
