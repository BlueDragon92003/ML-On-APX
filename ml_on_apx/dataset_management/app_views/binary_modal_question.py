from textual import on
from rich.text import Text
from textual.content import Content
from textual.widgets import Button
from textual.containers import VerticalGroup, VerticalScroll
from textual.app import ComposeResult
from textual.widget import Widget
from textual.screen import ModalScreen


class BinaryModalQuestion(ModalScreen[bool]):
    BINDINGS = [("escape", "exit", "Exit")]

    def __init__(
        self,
        *questions: Widget,
        title: str | None = None,
        subtitle: str | None = None,
        true_button_label: Content | Text | str | None = "Yes",
        false_button_label: Content | Text | str | None = "No",
    ):
        super().__init__()
        self._questions = questions
        self._title = title
        self._subtitle = subtitle
        self._true_button_label = true_button_label
        self._false_button_label = false_button_label

    def compose(self) -> ComposeResult:
        with VerticalGroup(classes="container", id="container"):
            yield VerticalScroll(*self._questions, classes="question")
            yield Button(self._true_button_label, variant="success", id="bmodal-true")
            yield Button(self._false_button_label, variant="error", id="bmodal-false")

    def on_mount(self):
        container = self.get_child_by_id("container")
        container.border_title = self._title
        container.border_subtitle = self._subtitle
        container.get_child_by_id("bmodal-true").focus()

    def action_exit(self):
        self.dismiss(False)

    @on(Button.Pressed)
    def handle_button_press(self, message: Button.Pressed):
        button_id = message.button.id
        match button_id:
            case "bmodal-true":
                self.dismiss(True)
            case "bmodal-false":
                self.dismiss(False)
