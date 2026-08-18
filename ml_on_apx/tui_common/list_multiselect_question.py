"""A question where the user selects from a list of options."""

from typing import ClassVar, Hashable, Iterable, TypeVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalGroup
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Button, SelectionList

from ml_on_apx.logging import log_call
from ml_on_apx.tui_common import _TUI, Popup

_LMSQ = "lmsq" @ _TUI

ListItem = TypeVar("ListItem", bound=Hashable)


class ListMultiselectQuestion(ModalScreen[set[ListItem]], Popup):
    """A question with a finite set of answers."""

    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("escape", "exit", "Cancel")
    ]

    def __init__(
        self,
        options: Iterable[tuple[str, ListItem, bool]],
        title: str | None = None,
        subtitle: str | None = None,
        save_button_label: Content | Text | str | None = "Save",
        exit_button_label: Content | Text | str | None = "Cancel",
    ) -> None:
        """Initialize a new ListSelectQuestion.

        Args:
            options (list[tuple[str, ListItem]]): A list of options the user can select
                from.
            title (str | None, optional): The title of the box. Defaults to None.
            subtitle (str | None, optional): The subtitle around the box. Defaults to
                None.
            save_button_label (Content | Text | str | None, optional): The label for the
                button that returns the selected items. Defaults to "Save".
            exit_button_label (Content | Text | str | None, optional): The label for
                the button that returns None. Defaults to "Cancel".

        """
        super().__init__()
        self._options = options
        self._title = title
        self._subtitle = subtitle
        self._save_button_label = save_button_label
        self._exit_button_label = exit_button_label

    def compose(self) -> ComposeResult:
        """Build the screen from its component widgets.

        Returns:
            ComposeResult: An iterator of widgets this screen incorporates.

        Yields:
            Iterator[ComposeResult]: A widget to incorporated.

        """
        with VerticalGroup(classes="container", id="container"):
            yield SelectionList[ListItem](
                *self._options, id="mslist-list", classes="question"
            )
            yield Button(self._save_button_label, variant="success", id="lmsq-save")
            yield Button(self._exit_button_label, variant="error", id="lmsq-exit")

    @log_call(action_type="mount" > _LMSQ)
    def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""
        container = self.get_child_by_id("container")
        container.border_title = self._title
        container.border_subtitle = self._subtitle
        container.get_child_by_id("mslist-list").focus()

    @log_call(action_type="exit" > _LMSQ)
    def action_exit(self) -> None:
        """Process the action `exit`."""
        self.dismiss(None)

    @on(Button.Pressed)
    @log_call(action_type="button" > _LMSQ)
    def handle_button_press(self, message: Button.Pressed) -> None:
        """Handle the Pressed event from a child button.

        Args:
            message (Button.Pressed): The event to handle.

        """
        button_id = message.button.id
        selected: list[ListItem] = self.get_widget_by_id(
            "mslist-list", SelectionList
        ).selected
        match button_id:
            case "lmsq-save":
                self.dismiss(set(selected))
            case "lmsq-exit":
                self.dismiss(None)
