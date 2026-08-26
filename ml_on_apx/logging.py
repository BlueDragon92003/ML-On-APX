"""Logging constants and helper functions."""

import datetime
import os
import re
from functools import cmp_to_key
from inspect import signature
from pathlib import Path
from typing import Awaitable, Callable, Concatenate, Iterable, ParamSpec, TypeVar

import eliot
from eliot import start_action

_R = TypeVar("_R")
_P = ParamSpec("_P")
_Q = ParamSpec("_Q")
_S = TypeVar("_S")

CallbackDecorator = Callable[
    [Callable[_Q, Awaitable[None]]],
    Callable[_Q, Awaitable[None]],
]


class MissingDecoratorArgError(Exception):
    """Raised when a decorated function does not accept a callback decorator."""


class IncludedMissingCallerArgError(Exception):
    """An included_caller_arg is missing from the function signature."""


class IncludedMissingCallbackArgError(Exception):
    """An included_callback_arg is missing from the function signature."""


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
            raise KeyError()
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
            raise KeyError()
        self._namespaces.add(namespace_name)
        self._names.add(namespace_name)
        return Namespace(self._path + ":" + namespace_name)

    def __str__(self) -> str:
        """Get the fully-qualified name for this namespace."""
        return self._path


_LOG = Namespace("log")
_LOG_SETUP = "setup" @ _LOG

_LOG_SETUP_DIRECTORY = "dir" > _LOG_SETUP


def log_call(
    action_type: str | None = None,
    include_args: list[str] | None = None,
    include_result: bool = True,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Typed wrapper for `eliot`'s log_call decorator."""
    return eliot.log_call(None, action_type, include_args, include_result)


def log_with_callback(
    action_type: str,
    include_caller_args: Iterable[str] | None = None,
    include_callback_args: Iterable[str] | None = None,
    decorator_arg_name: str = "callback",
) -> Callable[
    [Callable[Concatenate[_S, CallbackDecorator[_Q], _P], None]],
    Callable[Concatenate[_S, _P], None],
]:
    """Log this call and provide a callback decorator."""

    def caller_decorator(
        caller: Callable[Concatenate[_S, CallbackDecorator[_Q], _P], None],
    ) -> Callable[Concatenate[_S, _P], None]:
        caller_signature = signature(caller)
        if decorator_arg_name not in set(caller_signature.parameters):
            raise MissingDecoratorArgError
        if (include_caller_args is not None) and (
            extra := (set(include_caller_args) - set(caller_signature.parameters))
            - set(decorator_arg_name)
        ):
            raise IncludedMissingCallerArgError(extra)

        def caller_wrapper(slf: _S, *args: _P.args, **kwargs: _P.kwargs) -> None:
            # Use "None" to create a temporary fake action
            caller_args = caller_signature.bind(slf, None, *args, **kwargs).arguments
            caller_args.pop(decorator_arg_name)
            if "self" in caller_args:
                caller_args.pop("self")
            if include_caller_args is not None:
                caller_args = {x: caller_args[x] for x in include_caller_args}
            action = start_action(action_type=action_type, **caller_args)

            def callback_decorator(
                callback: Callable[_Q, Awaitable[None]],
            ) -> Callable[_Q, Awaitable[None]]:
                callback_signature = signature(callback)
                if (include_callback_args is not None) and (
                    extra := set(include_callback_args)
                    - set(callback_signature.parameters)
                ):
                    raise IncludedMissingCallbackArgError(extra)

                async def callback_wrapper(
                    *bargs: _Q.args, **bkwargs: _Q.kwargs
                ) -> None:
                    callback_args = callback_signature.bind(*bargs, **bkwargs).arguments
                    if include_callback_args is not None:
                        callback_args = {
                            x: callback_args[x] for x in include_callback_args
                        }
                    with action as ctx:
                        ctx.log(message_type=f"callback.{action_type}", **callback_args)
                        await callback(*bargs, **bkwargs)

                return callback_wrapper

            with action.context():
                caller(slf, callback_decorator, *args, **kwargs)

        return caller_wrapper

    return caller_decorator


@log_call(action_type="compare_files" > _LOG_SETUP)
def _compare_files(file1: Path, file2: Path) -> int:
    stem1 = file1.stem.split("-")
    stem2 = file2.stem.split("-")
    if len(stem1) == 3:  # noqa: PLR2004
        stem1.append("0")
    if len(stem2) == 3:  # noqa: PLR2004
        stem2.append("0")
    for i in range(4):
        if int(stem2[i]) != int(stem1[i]):
            return int(stem1[i]) - int(stem2[i])
    return int(os.path.getctime(file1) - os.path.getctime(file2))


@log_call(action_type="init" > _LOG, include_result=False)
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

    @log_call(action_type=_LOG_SETUP_DIRECTORY, include_result=False)
    def log_setup_directory() -> None:
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
                for td in to_delete:
                    os.unlink(td)
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
        if (log_file / "latest.log").exists():
            (log_file / "latest.log").unlink()
        os.symlink(
            target,
            log_file / "latest.log",
            target_is_directory=False,
        )

    if log_file.is_file():
        eliot.to_file(open(log_file, "ab" if append else "wb"))
    else:
        log_setup_directory()
