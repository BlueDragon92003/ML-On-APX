import uproot

import cluster_classification_dataset
import cluster

import unittest

class TestDataWrapper(unittest.TestCase):

    def test_data_wrapper__get_clusters(self):
        """
        Ensure that the cluster data has the expected shape.
        """
        tree = uproot.open("data/testing/test.root")["l1NtupleProducer/linkTree;1"]
        cluster_data = cluster_classification_dataset.ClusterClassificationDataset._ClusterClassificationDataset__get_clusters(tree)
        (_, cards, slr) = cluster_data.shape
        self.assertEqual(4, slr)
        self.assertEqual(24, cards)

    def test_data_wrapper__get_ecal(self):
        """
        Ensure that the ecal data has the expected shape.
        """
        tree = uproot.open("data/testing/test.root")["l1NtupleProducer/linkTree;1"]
        ecal_data = cluster_classification_dataset.ClusterClassificationDataset._ClusterClassificationDataset__get_ecal_towers(tree)
        (_, cards, slr) = ecal_data.shape
        self.assertEqual(4, slr)
        self.assertEqual(24, cards)

    def test_data_wrapper__get_hcal(self):
        """
        Ensure that the hcal data has the expected shape.
        """
        tree = uproot.open("data/testing/test.root")["l1NtupleProducer/linkTree;1"]
        hcal_data = cluster_classification_dataset.ClusterClassificationDataset._ClusterClassificationDataset__get_hcal_towers(tree)
        (_, cards, link) = hcal_data.shape
        self.assertEqual(4, link)
        self.assertEqual(24, cards)

    def test_data_wrapper__create_cluster_classification(self):
        """
        Ensure the creation of the dataset does not error.
        """
        cluster_classification_dataset.ClusterClassificationDataset(
            {('data/testing/test.root', cluster.ClusterType.BACKGROUND)}
            )

    # TODO create correctness tests
