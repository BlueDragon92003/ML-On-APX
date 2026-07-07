"""Logging constants and helper functions."""

import datetime
import os
import re
from functools import cmp_to_key
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import eliot
from eliot import ActionType, fields

_R = TypeVar("_R")
_P = ParamSpec("_P")


class Namespace(str):
    """The namespace of an action."""

    def __rmatmul__(self, name: str) -> "Namespace":
        """Specify a child namespace.

        Args:
            name (Namespace): The name of the child namespace.

        Returns:
            Namespace: The child namespace

        """
        return Namespace(self + ":" + name)


_LOG = Namespace("log")
_LOG_SETUP = "setup" @ _LOG

_LOG_SETUP_FILE = ActionType(
    action_type="file" @ _LOG_SETUP,
    startFields=fields(filepath=Path, append=bool),
    successFields=fields(),
    description="Setup logging to one file",
)

_LOG_SETUP_DIRECTORY = ActionType(
    action_type="dir" @ _LOG_SETUP,
    startFields=fields(filepath=Path, file_count=int),
    successFields=fields(),
    description="",
)


def _compare_files(file1: Path, file2: Path) -> int:
    stem1 = file1.stem.split("-")
    stem2 = file2.stem.split("-")
    for i in range(4):
        if int(stem2[i]) != int(stem1[i]):
            return int(stem2[i]) - int(stem1[i])
    return int(os.path.getctime(file2) - os.path.getctime(file1))


def log_call(
    action_type: str | None = None,
    include_args: list[str] | None = None,
    include_result: bool = True,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Typed wrapper for `eliot`'s log_call decorator."""
    return eliot.log_call(None, action_type, include_args, include_result)


@log_call(action_type="init" @ _LOG)
def initialize_file_logging(
    log_file: Path,
    append: bool = False,
    file_count: int = 0,
) -> None:
    """Initialize the logging system.

    Args:
        log_file (Path): A filepath to log to. If a directory, automatically creates and
            deletes log files with the name `{date}-{num}`.log. If a file, it appends or
            writes to that file.
        append (bool): If the filepath is a normal file, append to the file rather than
            overwriting it.
        file_count (int): The number of logs to save. If a log file must be deleted, the
            oldest is deleted. If this number is non-positive, no limits are made.
        console_log_level (LogLevel, optional): The level of message to log to the
            console/using textual's notification system. Defaults to
            `LogLevel.FOR_PRODUCTION`.

    """
    if log_file.is_file():
        with _LOG_SETUP_FILE(filepath=log_file, append=append):
            eliot.to_file(open(log_file, "ab" if append else "wb"))
    else:
        with _LOG_SETUP_DIRECTORY(filepath=log_file, file_count=file_count):
            if file_count > 0:
                log_files = list(
                    filter(
                        lambda path: re.fullmatch(
                            r"\d{4,}-\d\d-\d\d-\d+.log", path.name
                        ),
                        log_file.iterdir(),
                    )
                )
                if (diff := len(log_files) + 1 - file_count) > 0:
                    to_delete = sorted(log_files, key=cmp_to_key(_compare_files))[:diff]
                    for td in to_delete:
                        os.unlink(td)
            today = datetime.date.today()
            base = f"{today.year:04d}-{today.month:02d}-{today.day:02d}"
            stem = base
            count = 1
            while (log_file / f"{stem}.log").exists():
                stem = f"{base}-{count}"
                count += 1
            eliot.to_file(open(log_file / f"{stem}.log", "xb"))
            os.symlink(
                log_file / f"{stem}.log",
                log_file / "latest.log",
                target_is_directory=False,
            )
