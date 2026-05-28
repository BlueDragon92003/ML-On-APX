import pathlib
import unittest
from colour_runner.runner import ColourTextTestRunner

if __name__ == "__main__":
    loader = unittest.TestLoader()
    start_dir = pathlib.Path(__file__).parent
    suite = loader.discover(str(start_dir))
    runner = ColourTextTestRunner()
    runner.run(suite)