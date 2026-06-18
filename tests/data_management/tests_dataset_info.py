from pathlib import Path
from ml_on_apx.labelling import Labels, Label
from ml_on_apx.dataset_management.dataset_info import DatasetInfo
import unittest


class TestDatasetInfo(unittest.TestCase):
    def test_dataset_info__instantiation(self):
        labels = Labels([Label("a"), Label("c")])
        sources = [
            (Path("first"), Label("a")),
            (Path("second"), Label("a")),
            (Path("third"), Label("c")),
        ]
        DatasetInfo(labels, sources)

    def test_dataset_info__missing_labels(self):
        labels = Labels([Label("a")])
        sources = [
            (Path("first"), Label("a")),
            (Path("second"), Label("a")),
            (Path("third"), Label("c")),
        ]
        with self.assertRaises(ValueError):
            DatasetInfo(labels, sources)

    def test_dataset_info__get_labels(self):
        labels = Labels([Label("a"), Label("c")])
        sources = [
            (Path("first"), Label("a")),
            (Path("second"), Label("a")),
            (Path("third"), Label("c")),
        ]
        dsinfo = DatasetInfo(labels, sources)
        self.assertEqual(dsinfo.get_labels(), labels)

    def test_dataset_info__get_sources(self):
        labels = Labels([Label("a"), Label("c")])
        sources = [
            (Path("first"), Label("a")),
            (Path("second"), Label("a")),
            (Path("third"), Label("c")),
        ]
        dsinfo = DatasetInfo(labels, sources)
        expected = set([Path("first"), Path("second"), Path("third")])
        self.assertEqual(dsinfo.get_sources(), expected)

    def test_dataset_info__get_labeled_sources(self):
        labels = Labels([Label("a"), Label("c")])
        sources = [
            (Path("first"), Label("a")),
            (Path("second"), Label("a")),
            (Path("third"), Label("c")),
        ]
        dsinfo = DatasetInfo(labels, sources)
        expected = set([(Path("first"), 0), (Path("second"), 0), (Path("third"), 1)])
        self.assertEqual(dsinfo.get_labeled_sources(), expected)
