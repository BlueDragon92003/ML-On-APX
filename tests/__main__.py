from pathlib import Path
import pathlib
import unittest
from colour_runner.runner import ColourTextTestRunner

from ml_on_apx.cleverlogger import CleverLogger

if __name__ == "__main__":
    log_path = Path(__file__).parent.parent / "logs" / "tests"
    if not log_path.exists():
        log_path.mkdir()
    elif not log_path.is_dir():
        raise NotADirectoryError(f"Expected a directory at `{str(log_path)}`.")

    for file in log_path.iterdir():
        file.unlink()

    CleverLogger.file_log_level = "NONE"
    CleverLogger.console_log_level = "NONE"
    CleverLogger.log_file = log_path

    loader = unittest.TestLoader()
    start_dir = pathlib.Path(__file__).parent
    suite = loader.discover(str(start_dir))
    runner = ColourTextTestRunner(verbosity=2)
    runner.run(suite)
