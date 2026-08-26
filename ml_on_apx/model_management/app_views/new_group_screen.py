"""The screen for selecting a new group."""

from typing import ClassVar, Union

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import HorizontalGroup
from textual.reactive import reactive
from textual.screen import Screen
from textual.types import NoSelection
from textual.validation import Integer, Number, ValidationResult, Validator
from textual.widgets import Button, Input, Select
from textual.widgets import Label as TuiLabel

from ml_on_apx.model_management.stop_functions import StopFunction
from ml_on_apx.model_management.training_job import TrainingJob, TrainingJobBuilder

"""
1. Select a group (or none) to base this one from
    - If none was selected, specify a model type
2. Tweak/create the model
"""


class NewGroupScreen(Screen[TrainingJob]):
    """Creates a training job to create a new group."""

    BINDINGS: ClassVar[list[Union[tuple[str, str, str], Binding]]] = [
        ("esc", "cancel", "Cancel"),
        ("^s", "save", "Create"),
    ]

    _dataset_options: list[tuple[str, str]]
    _model_options: list[tuple[str, str]]

    dataset: reactive[str | None] = reactive(None)
    stop_fn: reactive[StopFunction | None] = reactive(None)
    lookback_dist: reactive[int] = reactive(3)
    batch_size: reactive[int] = reactive(1)
    checkpoint: reactive[int] = reactive(10)
    learn_rate: reactive[float] = reactive(1e-4)
    testing_set: reactive[str | None] = reactive(None)
    base_model: reactive[str | None] = reactive(None)

    class StopFunctionValidator(Validator):
        """Validates provided stop functions."""

        def validate(self, value: str) -> ValidationResult:
            """Check if a string is (mostly) valid lisp."""
            try:
                StopFunction(value)
            except IndexError:
                return self.failure("Improper syntax.")
            else:
                return self.success()

    def __init__(
        self, dataset_options: list[str], model_options: list[str], group_name: str
    ) -> None:
        """Initialize a new NewGroupScreen."""
        super(NewGroupScreen, self).__init__()
        self._dataset_options = [(x, x) for x in dataset_options]
        self._model_options = [(x, x) for x in model_options]
        self._group_name = group_name

    def compose(self) -> ComposeResult:
        """Build the screen from its component widgets."""
        with HorizontalGroup():
            yield TuiLabel("Dataset: ")
            yield Select(self._dataset_options, id="dataset")
        with HorizontalGroup():
            yield TuiLabel("Stop Function: ")
            yield Input(
                placeholder="( quote () )",
                validators=[self.StopFunctionValidator()],
                validate_on=["changed"],
                id="stop-fn",
            )
            # TODO help button or markdown component
        with HorizontalGroup():
            yield TuiLabel("Stop Function Lookback Distance: ")
            yield Input(
                value="3",
                validators=[
                    Integer(minimum=1, failure_description="Must be a positive integer")
                ],
                validate_on=["changed"],
                id="lookback",
            )
        with HorizontalGroup():
            yield TuiLabel("Batch Size: ")
            yield Input(
                value="1",
                validators=[
                    Integer(minimum=1, failure_description="Must be a positive integer")
                ],
                validate_on=["changed"],
                id="batch-size",
            )
        with HorizontalGroup():
            yield TuiLabel("Epochs between checkpoints: ")
            yield Input(
                value="10",
                validators=[
                    Integer(minimum=1, failure_description="Must be a positive integer")
                ],
                validate_on=["changed"],
                id="checkpoint",
            )

        with HorizontalGroup():
            yield TuiLabel("Learning rate: ")
            yield Input(
                value="10",
                validators=[
                    Number(
                        minimum=5e-324,
                        failure_description="Must be a positive real number",
                    )
                ],
                validate_on=["changed"],
                id="learn-rate",
            )
        with HorizontalGroup():
            yield TuiLabel("Dataset for use during Testing (optional): ")
            yield Select(self._dataset_options, id="testing-dataset")
        with HorizontalGroup():
            yield TuiLabel("Model to use as a base: ")
            yield Select(self._model_options, id="base-model")
        with HorizontalGroup():
            yield Button("Cancel", variant="default", id="cancel-btn")
            yield Button("Create", variant="primary", id="create-btn")

    def on_mount(self) -> None:
        """Finish setup of the screen once it is attached to the DOM."""

    @on(Select.Changed, "#dataset")
    def handle_dataset_changed(self, message: Select.Changed) -> None:
        """Set the training dataset when the selection changes."""
        match message.value:
            case x if type(x) is NoSelection:
                self.dataset = None
            case x if type(x) is str:
                self.dataset = x

    @on(Select.Changed, "#testing-dataset")
    def handle_test_dataset_changed(self, message: Select.Changed) -> None:
        """Set the testing dataset when the selection changes."""
        match message.value:
            case x if type(x) is NoSelection:
                self.testing_set = None
            case x if type(x) is str:
                self.testing_set = x

    @on(Select.Changed, "#base-model")
    def handle_base_changed(self, message: Select.Changed) -> None:
        """Set the base model when the selection changes."""
        match message.value:
            case x if type(x) is NoSelection:
                self.base_model = None
            case x if type(x) is str:
                self.base_model = x

    @on(Input.Changed, "#stop-fn")
    def handle_stop_fn_changed(self, message: Input.Changed) -> None:
        """Set the stop function when the input changes."""
        if message.validation_result is not None:
            if message.validation_result.is_valid:
                try:
                    self.stop_fn = StopFunction(message.value)
                    # Should not error- error handling happens in the validator
                except IndexError:
                    self.stop_fn = None

    @on(Input.Changed, "#lookback")
    def handle_lookback_changed(self, message: Input.Changed) -> None:
        """Set the lookback depth when the input changes."""
        if message.validation_result is not None and message.validation_result.is_valid:
            self.lookback_dist = int(message.value)

    @on(Input.Changed, "#batch-size")
    def handle_batch_size_changed(self, message: Input.Changed) -> None:
        """Set the batch size when the input changes."""
        if message.validation_result is not None and message.validation_result.is_valid:
            self.batch_size = int(message.value)

    @on(Input.Changed, "#checkpoint")
    def handle_checkpoint_changed(self, message: Input.Changed) -> None:
        """Set the checkpoint rate when the input changes."""
        if message.validation_result is not None and message.validation_result.is_valid:
            self.checkpoint = int(message.value)

    @on(Input.Changed, "#learn-rate")
    def handle_learn_rate_changed(self, message: Input.Changed) -> None:
        """Set the learning rate when the input changes."""
        if message.validation_result is not None and message.validation_result.is_valid:
            self.learn_rate = float(message.value)

    @on(Button.Pressed, "#cancel-btn")
    def action_cancel(self) -> None:
        """Cancel the creation of the training job."""
        self.dismiss(None)

    @on(Button.Pressed, "#create-btn")
    def action_save(self) -> None:
        """Save the training job."""
        builder = TrainingJobBuilder()
        builder.group_name(self._group_name)
        if self.dataset is None:
            self.app.notify("Please select a dataset.", severity="error")
            return
        builder.dataset(self.dataset)
        if self.stop_fn is None:
            self.app.notify("Please provide a lisp stop function.", severity="error")
            return
        builder.stop_function(self.stop_fn)
        builder.lookback_distance(self.lookback_dist)
        builder.batch_size(self.batch_size)
        builder.checkpoint_rate(self.checkpoint)
        builder.learning_rate(self.learn_rate)
        builder.testing_dataset(self.testing_set)
        builder.base_model_name(self.base_model)
        try:
            job = builder.build()
        except ValueError:
            self.app.notify(
                "Please ensure all fields have valid values.", severity="warning"
            )
        except TypeError:
            self.app.notify(
                "Please ensure both required fields are proided.", severity="warning"
            )
        else:
            self.dismiss(job)
