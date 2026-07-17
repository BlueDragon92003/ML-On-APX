"""Logging constants and helper functions."""

import datetime
import os
import re
from functools import cmp_to_key
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import eliot
from eliot import Action, ActionType, fields

_R = TypeVar("_R")
_P = ParamSpec("_P")


class Namespace:
    """The namespace of an action."""

    def __init__(self, name: str) -> None:
        """Create a new namespace.

        Args:
            name (str): The name of the namespace.

        """
        self._path = name
        self._namespaces: set[str] = set()
        self._names: set[str] = set()

    def __lt__(self, name: str) -> str:
        """Add a new non-namespace name to the namespace.

        Args:
            name (str): The name to add

        Raises:
            KeyError: If the name was already defined.

        Returns:
            str: The fully-qualified name added

        """
        if name in self._names:
            raise KeyError("Name already defined!")
        self._names.add(name)
        return self._path + ":" + name

    def __rmatmul__(self, namespace_name: str) -> "Namespace":
        """Specify a child namespace.

        Args:
            namespace_name (str): The name of the child namespace.

        Raises:
            KeyError: If the name was already defined.

        Returns:
            Namespace: The child namespace

        """
        if namespace_name in self._names:
            raise KeyError("Name already defined!")
        self._namespaces.add(namespace_name)
        self._names.add(namespace_name)
        return Namespace(self._path + ":" + namespace_name)

    def __str__(self) -> str:
        """Get the fully-qualified name for this namespace."""
        return self._path


_LOG = Namespace("log")
_LOG_SETUP = "setup" @ _LOG

_LOG_SETUP_FILE = ActionType(
    action_type="file" @ _LOG_SETUP,
    startFields=fields(filepath=str, append=bool),
    successFields=fields(),
    description="Setup logging to one file",
)

_LOG_SETUP_DIRECTORY = ActionType(
    action_type="dir" @ _LOG_SETUP,
    startFields=fields(filepath=str, file_count=int),
    successFields=fields(newfile=str, deleted=list),
    description="",
)


def log_call(
    action_type: str | None = None,
    include_args: list[str] | None = None,
    include_result: bool = True,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Typed wrapper for `eliot`'s log_call decorator."""
    return eliot.log_call(None, action_type, include_args, include_result)


# @log_call("compare_files" @ _LOG_SETUP)
def _compare_files(file1: Path, file2: Path) -> int:
    stem1 = file1.stem.split("-")
    stem2 = file2.stem.split("-")
    if len(stem1) == 3:
        stem1.append("0")
    if len(stem2) == 3:
        stem2.append("0")
    for i in range(4):
        if int(stem2[i]) != int(stem1[i]):
            return int(stem1[i]) - int(stem2[i])
    return int(os.path.getctime(file1) - os.path.getctime(file2))


@log_call(action_type="init" > _LOG)
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
        with _LOG_SETUP_DIRECTORY(filepath=log_file, file_count=file_count) as action:
            assert type(action) is Action
            log_files = list(
                filter(
                    lambda path: re.fullmatch(r"\d{4,}-\d\d-\d\d-\d+.log", path.name),
                    log_file.iterdir(),
                )
            )
            log_files = sorted(log_files, key=cmp_to_key(_compare_files))
            if file_count > 0:
                if (diff := len(log_files) + 1 - file_count) > 0:
                    to_delete = log_files[:diff]
                    log_files = log_files[diff:]
                    action.add_success_fields(deleted=to_delete)
                    for td in to_delete:
                        os.unlink(td)
                else:
                    action.add_success_fields(deleted=None)
            newest = log_files[-1].stem if len(log_files) != 0 else None
            today = datetime.date.today()
            base = f"{today.year:04d}-{today.month:02d}-{today.day:02d}"
            if newest:
                split = newest.split("-")
                if today != datetime.date(int(split[0]), int(split[1]), int(split[2])):
                    count = 0
                else:
                    count = int(split[3])
            else:
                count = 0
            stem = f"{base}-{count}"
            target = log_file / f"{stem}.log"
            while target.exists():
                count += 1
                stem = f"{base}-{count}"
                target = log_file / f"{stem}.log"
            eliot.to_file(open(target, "xb"))
            (log_file / "latest.log").unlink()
            os.symlink(
                target,
                log_file / "latest.log",
                target_is_directory=False,
            )
            action.add_success_fields(newfile=str(target))
