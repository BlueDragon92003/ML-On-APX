import unittest
from colour_runner.runner import ColourTextTestRunner

from tests.tests_dataset_subset import TestDatasetSubset
from tests.tests_data_wrapper import TestDataWrapper

if __name__ == "__main__":
    unittest.main(testRunner=ColourTextTestRunner)
