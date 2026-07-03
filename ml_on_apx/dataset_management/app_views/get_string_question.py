from typing import Callable
from textual import on
from textual.widgets import Input
from textual.containers import VerticalGroup
from textual.app import ComposeResult
from textual.screen import ModalScreen


class GetStringQuestion(ModalScreen[str]):
    BINDINGS = [("escape", "exit", "Cancel")]

    def __init__(
        self,
        validator: Callable[[str], bool] = lambda x: True,
        title: str | None = None,
        subtitle: str | None = None,
    ):
        super().__init__()
        self._validator = validator
        self._title = title
        self._subtitle = subtitle

    def compose(self) -> ComposeResult:
        with VerticalGroup(classes="container", id="container"):
            yield Input(id="slist-list")

    def on_mount(self):
        container = self.get_child_by_id("container")
        container.border_title = self._title
        container.border_subtitle = self._subtitle
        container.get_child_by_id("slist-list").focus()

    def action_exit(self):
        self.dismiss(None)

    @on(Input.Submitted)
    def handle_input_submission(self, message: Input.Submitted):
        if self._validator(message.value):
            self.dismiss(message.value)
        else:
            self.app.notify("Not a valid input!")
