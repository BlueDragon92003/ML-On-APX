"""Tests for the labelling module."""

import unittest

from eliot.testing import capture_logging

from ml_on_apx.labelling import Label, Labels


class TestLabels(unittest.TestCase):
    """Tests for the Labelling module object."""

    @capture_logging
    def test_labelling__test_label_eq(self) -> None:
        """Test that two identical labels are equal."""
        label1 = Label("a")
        label2 = Label("a")
        self.assertEqual(label1, label2)

    @capture_logging
    def test_labelling__instantiation(self) -> None:
        """Test the creation of a valid Labels object does not error."""
        Labels({Label("a"), Label("b")})

    @capture_logging
    def test_labelling__contains(self) -> None:
        """Test that the __contains__ magic method works as expected."""
        labels = Labels({Label("a"), Label("b")})
        self.assertTrue(Label("a") in labels)
        self.assertTrue(Label("b") in labels)
        self.assertFalse(Label("c") in labels)

    @capture_logging
    def test_labelling__get_item(self) -> None:
        """Test that the __getitem__ magic method works as expeced."""
        labels = Labels({Label("a"), Label("b")})
        self.assertEqual(0, labels[Label("a")])
        self.assertEqual(1, labels[Label("b")])
        with self.assertRaises(KeyError):
            labels[Label("c")]

    @capture_logging
    def test_labelling__test_labels_eq(self) -> None:
        """Test that the __eq__ magic method works as ecpected."""
        labels1 = Labels({Label("a"), Label("b")})
        labels2 = Labels({Label("b"), Label("a")})
        self.assertEqual(labels1, labels2)
