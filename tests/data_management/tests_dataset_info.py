"""Tests for the DatasetInfo class."""

import unittest
from pathlib import Path

from ml_on_apx.dataset_management.dataset_info import DatasetInfo
from ml_on_apx.labelling import Label, Labels


class TestDatasetInfo(unittest.TestCase):
    """Tests for the DatasetInfo class."""

    def test_dataset_info__instantiation(self) -> None:
        """Test that the creation of a valid DatasetInfo object does not error."""
        labels = Labels([Label("a"), Label("c")])
        sources = [
            (Path("first"), Label("a")),
            (Path("second"), Label("a")),
            (Path("third"), Label("c")),
        ]
        DatasetInfo(labels, sources)

    def test_dataset_info__missing_labels(self) -> None:
        """Test that the creation of a DatasetInfo missing a label object errors."""
        labels = Labels([Label("a")])
        sources = [
            (Path("first"), Label("a")),
            (Path("second"), Label("a")),
            (Path("third"), Label("c")),
        ]
        with self.assertRaises(ValueError):
            DatasetInfo(labels, sources)

    def test_dataset_info__get_labels(self) -> None:
        """Test that the get_lables function works as expected."""
        labels = Labels([Label("a"), Label("c")])
        sources = [
            (Path("first"), Label("a")),
            (Path("second"), Label("a")),
            (Path("third"), Label("c")),
        ]
        dsinfo = DatasetInfo(labels, sources)
        self.assertEqual(dsinfo.labels, labels)

    def test_dataset_info__get_sources(self) -> None:
        """Test that the get_sources function works as expected."""
        labels = Labels([Label("a"), Label("c")])
        sources = [
            (Path("first"), Label("a")),
            (Path("second"), Label("a")),
            (Path("third"), Label("c")),
        ]
        dsinfo = DatasetInfo(labels, sources)
        expected = {Path("first"), Path("second"), Path("third")}
        self.assertEqual(dsinfo.sources, expected)

    def test_dataset_info__get_labeled_sources(self) -> None:
        """Test that the get_labeled_sources function works as expected."""
        labels = Labels([Label("a"), Label("c")])
        sources = [
            (Path("first"), Label("a")),
            (Path("second"), Label("a")),
            (Path("third"), Label("c")),
        ]
        dsinfo = DatasetInfo(labels, sources)
        expected = {(Path("first"), 0), (Path("second"), 0), (Path("third"), 1)}
        self.assertEqual(dsinfo.numbered_sources, expected)
