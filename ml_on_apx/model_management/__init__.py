"""Manage ML models."""

from ml_on_apx.logging import Namespace

_MODEL = Namespace("model")
_TUI = "tui" @ _MODEL
_APP = "app" @ _TUI
