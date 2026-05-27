import unittest
from colour_runner.runner import ColourTextTestRunner

from cluster_classification.tests.tests_cluster_classification_dataset \
    import TestClusterClassificationDataset
from cluster_classification.tests.tests_dataset_subset import TestDatasetSubset

if __name__ == "__main__":
    unittest.main(testRunner=ColourTextTestRunner)
