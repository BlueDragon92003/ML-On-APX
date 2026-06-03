import pathlib
from datetime import datetime, date
from enum import Enum
from typing import Any, Mapping
import logging
import os

logging.addLevelName(2, "HYPERTRACE")
logging.addLevelName(5, "TRACE")
logging.addLevelName(45, "CRITICAL")


class RecordShape(Enum):
    """Represents the "shape" of the record should use"""

    OPEN = 0
    """'Opens' a record, which will be matched with a closing record"""
    CLOSE = 1
    """'Closes' an 'opened' record."""
    SINGLE = 2
    """Standalone Record"""


class XMLFormatter(logging.Formatter):
    """Formats the log record for printing to the log file.

    Overrides the default format method.
    """

    def __init__(self):
        super(XMLFormatter, self).__init__()

    # @override
    def format(self, record: logging.LogRecord) -> str:
        time: float = record.created
        name: str | None = record.msg
        location: str = record.name
        level: str = record.levelname
        process: int | None = record.process
        thread: int | None = record.thread
        # Additional arguments, will always be a dict via control over the actual log function.
        args: Mapping[str, Any] | None = record.arguments  # ty: ignore[unresolved-attribute] Added via extra
        shape: RecordShape = record.type  # ty: ignore[unresolved-attribute] Added via extra
        title: str | None = record.title  # ty: ignore[unresolved-attribute] Added via extra

        element = f"{level}:{location}"

        if name is not None:
            element += f":{name}"

        if thread is not None or process is not None:
            element += "--"
            if process is not None:
                element += to_base64(process)
            if thread is not None:
                element += ":" + to_base64(thread)

        additional = ""
        if title is not None:
            additional += ' :title="' + title.format(name=str(name)) + '"'
        if args is not None:
            for key, value in args.items():
                additional += " " + key + '="' + str(value) + '"'

        start_tag = "<"
        body = f'{element} :timestamp="{datetime.fromtimestamp(time).isoformat()}"{additional}'
        end_tag = ">"

        if shape is RecordShape.CLOSE:
            start_tag = start_tag + "/"
        if shape is RecordShape.SINGLE:
            end_tag = "/" + end_tag

        return f"{start_tag}{body}{end_tag}"


class ConsoleFormatter(logging.Formatter):
    """Formats the log record for printing to the terminal.

    Overrides the default format method.
    """

    def __init__(self):
        super(ConsoleFormatter, self).__init__()

    # @override
    def format(self, record: logging.LogRecord) -> str:
        name: str | None = record.msg
        location: str = record.name
        level: str = record.levelname
        process: int | None = record.process
        thread: int | None = record.thread
        # Additional arguments, will always be a dict via control over the actual log function.
        args: Mapping[str, Any] | None = record.arguments  # ty: ignore[unresolved-attribute] Added via extra
        title: str | None = record.title  # ty: ignore[unresolved-attribute] Added via extra

        if thread is not None or process is not None:
            location += ":"
            if process is not None:
                location += str(process)
            if thread is not None:
                location += ":" + str(thread)

        title = str(title).format(name=str(name))

        out = f"{level} at {location}: {title}"
        if args is not None and len(args) > 0:
            out += "Information:"
            for key, value in args.items():
                out += f"\n\t{key}: {value}"
        return out + "\n"


class CleverLogger:
    """A less-basic extention to the default python logger."""

    TRACE = logging.DEBUG - 5  # 5
    DEBUG = logging.DEBUG  # 10
    INFO = logging.INFO  # 20
    WARN = logging.WARN  # 30
    ERROR = logging.ERROR  # 40
    CRITICAL = logging.FATAL - 5  # 45
    FATAL = logging.FATAL  # 50

    major_process: str
    minor_process: str

    log_file: pathlib.Path | None = None

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

        file_log_level = os.environ.get(
            "FILE_LOG_LEVEL", os.environ.get("LOG_LEVEL", "INFO")
        ).upper()

        stream_log_level = os.environ.get(
            "CONSOLE_LOG_LEVEL", os.environ.get("LOG_LEVEL", "ERROR")
        ).upper()

        if CleverLogger.log_file is None:
            filestring = str(date.today())
            log_number = 0
            file = pathlib.Path(f"./logs/{filestring}.log.xml")
            while os.path.isfile(file):
                log_number += 1
                file = pathlib.Path(f"./logs/{filestring}-{log_number}.log.xml")
            CleverLogger.log_file = file
        file_handler = logging.FileHandler(CleverLogger.log_file)
        # os.symlink(file, './logs/latest.log.xml')
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(XMLFormatter())

        console_handler = logging.StreamHandler()
        console_handler.setLevel(stream_log_level)
        console_handler.setFormatter(ConsoleFormatter())

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    # TRACE
    def log_preloop(self, loop_name_and_type: str):
        self.logger.log(
            self.TRACE,
            loop_name_and_type,
            extra={
                "type": RecordShape.OPEN,
                "title": "Starting loop `{name}`...",
                "arguments": None,
            },
        )

    def log_postloop(self, loop_name_and_type: str):
        self.logger.log(
            self.TRACE,
            loop_name_and_type,
            extra={
                "type": RecordShape.CLOSE,
                "title": "Loop `{name}` finished.",
                "arguments": None,
            },
        )

    def log_iteration_head(self, **kwargs: Any):
        self.logger.log(
            self.TRACE,
            "iteration",
            extra={"type": RecordShape.OPEN, "title": "Iteration", "arguments": kwargs},
        )

    def log_iteration_tail(self):
        self.logger.log(
            self.TRACE,
            "iteration",
            extra={"type": RecordShape.CLOSE, "title": "Iteration", "arguments": None},
        )

    def log_iteration_exit(self, type: str):
        self.logger.log(
            self.TRACE,
            type,
            extra={"type": RecordShape.SINGLE, "title": "{name}", "arguments": None},
        )

    def log_micro_function(
        self, function_name: str, exit_type: str | None = None, **kwargs: Any
    ):
        """Trace travel through very simple functions for deep debugging."""
        self.logger.log(
            self.TRACE,
            function_name,
            extra={
                "type": RecordShape.SINGLE,
                "title": "`{name}` executed (" + str(exit_type) + ").",
                "arguments": kwargs,
            },
        )

    def log_enter_function(self, function_name: str, **kwargs: Any):
        """Trace entrance into functions for deep debugging."""
        self.logger.log(
            self.TRACE,
            function_name,
            extra={
                "type": RecordShape.OPEN,
                "title": "`{name}` called.",
                "arguments": kwargs,
            },
        )

    def log_exit_function(self, function_name: str):
        """Trace exit from functions for deep debugging."""
        self.logger.log(
            self.TRACE,
            function_name,
            extra={
                "type": RecordShape.CLOSE,
                "title": "`{name}` exited.",
                "arguments": None,
            },
        )

    def log_function_exit_type(self, exit_type: str, **kwargs: Any):
        """Trace exit information from the function."""
        self.logger.log(
            self.TRACE,
            exit_type,
            extra={
                "type": RecordShape.SINGLE,
                "title": "Exited via {name}.",
                "arguments": kwargs,
            },
        )

    # DEBUG
    def log_start_load_module(self, **kwargs: Any):
        """Log module loads."""
        self.logger.log(
            self.DEBUG,
            None,
            extra={
                "type": RecordShape.OPEN,
                "title": "Loading this module...",
                "arguments": kwargs,
            },
        )

    def log_end_load_module(
        self,
    ):
        """Log module loads."""
        self.logger.log(
            self.DEBUG,
            None,
            extra={
                "type": RecordShape.CLOSE,
                "title": "Loaded this module.",
                "arguments": None,
            },
        )

    def log_start_minor_process(self, minor_process_name: str, **kwargs: Any):
        """Log processes started by major processes."""
        self.logger.log(
            self.DEBUG,
            minor_process_name,
            extra={
                "type": RecordShape.OPEN,
                "title": "Starting `{name}`...",
                "arguments": kwargs,
            },
        )

    def log_end_minor_process(
        self,
        minor_process: str,
    ):
        """Log processes started by major processes."""
        self.logger.log(
            self.DEBUG,
            minor_process,
            extra={
                "type": RecordShape.CLOSE,
                "title": "Finished `{name}`.",
                "arguments": None,
            },
        )

    def log_variables(self, **kwargs: Any):
        """Log debug information for variables in key places."""
        self.logger.log(
            self.DEBUG,
            None,
            extra={
                "type": RecordShape.SINGLE,
                "title": "Debug Variables.",
                "arguments": kwargs,
            },
        )

    # INFO
    def log_notice(self, notice: str, **kwargs: Any):
        """Log general information pertinant to the execution of the program."""
        self.logger.log(
            self.INFO,
            None,
            extra={"type": RecordShape.SINGLE, "title": notice, "arguments": kwargs},
        )

    def log_start_major_process(self, major_process: str, **kwargs: Any):
        """Log processes started by the main function."""
        self.logger.log(
            self.INFO,
            major_process,
            extra={
                "type": RecordShape.OPEN,
                "title": "Starting `{name}`...",
                "arguments": kwargs,
            },
        )

    def log_end_major_process(self, major_process: str):
        """Log processes started by the main function."""
        self.logger.log(
            self.INFO,
            major_process,
            extra={
                "type": RecordShape.CLOSE,
                "title": "Finished `{name}`.",
                "arguments": None,
            },
        )

    # WARN
    def log_warning(self, warning: str, **kwargs: Any):
        """Log unexpected situations that do not break the code."""
        self.logger.log(
            self.WARN,
            None,
            extra={"type": RecordShape.SINGLE, "title": warning, "arguments": kwargs},
        )

    # ERROR
    def log_error(self, minor_process: str, error: str, **kwargs: Any):
        """Log unexpected situations that cause a minor process to fail."""
        self.logger.log(
            self.ERROR,
            minor_process,
            extra={
                "type": RecordShape.SINGLE,
                "title": "Process `{name}` encountered an error: " + error,
                "arguments": kwargs,
            },
        )

    # CRITICAL
    def log_critical(self, major_process: str, error: str, **kwargs: Any):
        """Log unexpected situations that cause a major process to fail."""
        self.logger.log(
            self.ERROR,
            major_process,
            extra={
                "type": RecordShape.SINGLE,
                "title": "Process `{name}` encountered an error: " + error,
                "arguments": kwargs,
            },
        )

    # FATAL
    def log_fatal(self, deathrattle: str, **kwargs: Any):
        """Log unexpected situations that cause the program to fail."""
        self.logger.log(
            self.ERROR,
            None,
            extra={
                "type": RecordShape.SINGLE,
                "title": deathrattle,
                "arguments": kwargs,
            },
        )


def to_base64(n):
    """Converts an integer to base 64 because thread IDs were too long"""
    digits = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
    nk = ""
    while n:
        nk += digits[n % 64]
        n //= 64
    return nk[::-1]
