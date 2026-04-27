import uproot

import cluster_classification_dataset

import unittest

class TestDataWrapper(unittest.TestCase):

    def test_data_wrapper__get_clusters(self):
        """
        Ensure that the cluster data has the expected shape.
        """
        tree = uproot.open("data/testing/test.root")["l1NtupleProducer/linkTree;1"]
        cluster_data = cluster_classification_dataset.ClusterClassificationDataset.get_clusters(tree)

        (len, cards, slr) = cluster_data.shape
        self.assertEqual(4, slr)
        self.assertEqual(24, cards)