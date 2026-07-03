from textual.types import NoSelection
from typing import TypeVar
from textual import on
from textual.widgets import Select
from textual.containers import VerticalGroup
from textual.app import ComposeResult
from textual.screen import ModalScreen

ListItem = TypeVar("ListItem")


class ListSelectQuestion(ModalScreen[ListItem]):
    BINDINGS = [("escape", "exit", "Cancel")]

    def __init__(
        self,
        options: list[tuple[str, ListItem]],
        title: str | None = None,
        subtitle: str | None = None,
    ):
        super().__init__()
        self._options = options
        self._title = title
        self._subtitle = subtitle

    def compose(self) -> ComposeResult:
        with VerticalGroup(classes="container", id="container"):
            yield Select[ListItem](self._options, id="slist-list")

    def on_mount(self):
        container = self.get_child_by_id("container")
        container.border_title = self._title
        container.border_subtitle = self._subtitle
        container.get_child_by_id("slist-list").focus()

    def action_exit(self):
        self.dismiss(None)

    @on(Select.Changed)
    def handle_selection(self, message: Select.Changed):
        assert type(message) is Select[ListItem].Changed
        value: ListItem | NoSelection = message.value  # ty: ignore[invalid-assignment]
        if type(value) is not NoSelection:
            self.dismiss(value)  # ty: ignore[invalid-argument-type]
        else:
            self.dismiss(None)
