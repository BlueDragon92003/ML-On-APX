"""The entry point for running the tests."""

import pathlib
import unittest
from pathlib import Path

from colour_runner.runner import ColourTextTestRunner

from ml_on_apx.logging import initialize_file_logging

if __name__ == "__main__":
    log_path = Path(__file__).parent.parent / "logs" / "tests"
    if not log_path.exists():
        log_path.mkdir()
    elif not log_path.is_dir():
        raise NotADirectoryError(f"Expected a directory at `{log_path!s}`.")

    for file in log_path.iterdir():
        file.unlink()

    initialize_file_logging(log_path, file_count=10)

    loader = unittest.TestLoader()
    start_dir = pathlib.Path(__file__).parent
    suite = loader.discover(str(start_dir))
    runner = ColourTextTestRunner(verbosity=2)
    runner.run(suite)
