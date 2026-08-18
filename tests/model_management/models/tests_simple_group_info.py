"""Tests for the SimpleGroupInfo class."""

import unittest

from ml_on_apx.labelling import Labels
from ml_on_apx.model_management.models.simple_model.simple_info import (
    InputLayerNoActivationError,
    SimpleGroupInfo,
)


class TestsSimpleGroupInfo(unittest.TestCase):
    """Tests for the GroupInfo class."""

    # All features same as input
    # labels same as input
    # initial features empty
    # layers length starts at 2
    def test_group_info__initialization(self) -> None:
        """Test that the GroupInfo object is insatiated correctly."""
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        self.assertEqual(labels, group_info.labels)
        self.assertEqual(features, group_info.all_features)
        self.assertEqual(0, len(group_info.features))
        self.assertEqual(2, group_info.layer_count)

    # Enable feaure works
    def test_group_info__enable_feature(self) -> None:
        """Test that enabling a feature works correctly."""
        feature = "alpha"
        labels = Labels("a", "b", "c")
        features = [feature, "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        group_info.enable_feature(feature)
        self.assertEqual(1, len(group_info.features))
        self.assertIn(feature, group_info.features)

    # disable feature works
    def test_group_info__disable_feature(self) -> None:
        """Test that disabling a feature works correctly."""
        feature = "alpha"
        labels = Labels("a", "b", "c")
        features = [feature, "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        group_info.enable_feature(feature)
        group_info.disable_feature(feature)
        self.assertEqual(0, len(group_info.features))
        self.assertNotIn(feature, group_info.features)

    # enable unavailable feature errors
    def test_group_info__enable_missing_feature(self) -> None:
        """Test that enabling a non-existant feature errors."""
        feature = "jeff"
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        with self.assertRaises(ValueError):
            group_info.enable_feature(feature)

    # disable unavailable feature errors
    def test_group_info__disable_missing_feature(self) -> None:
        """Test that disabling a non-existant feature errors."""
        feature = "jeff"
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        with self.assertRaises(ValueError):
            group_info.disable_feature(feature)

    # enable enabled feature silently "fails"
    def test_group_info__enable_enabled_feature(self) -> None:
        """Test that enabling an already-enabled feature does nothing."""
        feature = "alpha"
        labels = Labels("a", "b", "c")
        features = [feature, "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        group_info.enable_feature(feature)
        self.assertEqual(1, len(group_info.features))
        self.assertIn(feature, group_info.features)
        group_info.enable_feature(feature)
        self.assertEqual(1, len(group_info.features))
        self.assertIn(feature, group_info.features)

    # disable disabled feature silently "fails"
    def test_group_info__disable_disabled_feature(self) -> None:
        """Test that enabling an already-enabled feature does nothing."""
        feature = "alpha"
        labels = Labels("a", "b", "c")
        features = [feature, "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        group_info.enable_feature(feature)
        group_info.disable_feature(feature)
        self.assertEqual(0, len(group_info.features))
        self.assertNotIn(feature, group_info.features)
        group_info.disable_feature(feature)
        self.assertEqual(0, len(group_info.features))
        self.assertNotIn(feature, group_info.features)

    # add layer below works
    def test_group_info__insert_layer_below(self) -> None:
        """Test that insert_layer_below adds a layer below the target."""
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        group_info.insert_layer("aleph", 13)
        self.assertEqual(3, group_info.layer_count)
        self.assertEqual("aleph", group_info.get_layer_activation(1))
        group_info.insert_layer("bet", 7)
        self.assertEqual(4, group_info.layer_count)
        self.assertEqual("aleph", group_info.get_layer_activation(1))
        self.assertEqual("bet", group_info.get_layer_activation(2))

    # get layer size works for input layer
    def test_group_info__get_size_input(self) -> None:
        """Test that get_layer_size functions on the input layer."""
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        self.assertEqual(len(group_info.features), group_info.get_layer_size(0))
        group_info.enable_feature("alpha")
        self.assertEqual(len(group_info.features), group_info.get_layer_size(0))
        group_info.enable_feature("beta")
        self.assertEqual(len(group_info.features), group_info.get_layer_size(0))

    # get layer size works for output layer
    def test_group_info__get_size_output(self) -> None:
        """Test that get_layer_size functions on the output layer."""
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        self.assertEqual(len(labels), group_info.get_layer_size(1))

    # get layer size works for hidden layers
    def test_group_info__get_size_hidden(self) -> None:
        """Test that get_layer_size functions on a hidden layer."""
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        group_info.insert_layer("aleph", 13)
        self.assertEqual(13, group_info.get_layer_size(1))

    # get layer size works for oob layers
    def test_group_info__get_size_out_of_bounds(self) -> None:
        """Test that get_layer_size functions on an out-of-bounds layer."""
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        with self.assertRaises(IndexError):
            group_info.get_layer_size(13)
        with self.assertRaises(IndexError):
            group_info.get_layer_size(-5)

    # get layer activation works on output layer
    def test_group_info__get_activation_output(self) -> None:
        """Test that get_layer_activation works on the output layer."""
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        self.assertEqual(
            SimpleGroupInfo.DEFAULT_ACTIVATION, group_info.get_layer_activation(1)
        )

    # get layer activation errors on input layer
    def test_group_info__get_activation_input(self) -> None:
        """Test that get_layer_activation errors on the input layer."""
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        with self.assertRaises(InputLayerNoActivationError):
            group_info.get_layer_activation(0)

    # get layer activation works on hidden layer
    def test_group_info__get_activation_hidden(self) -> None:
        """Test that get_layer_activation works on hidden layers."""
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        group_info.insert_layer("aleph", 13)
        self.assertEqual("aleph", group_info.get_layer_activation(1))

    # get layer activation errors on oob layer
    def test_group_info__get_activation_oob(self) -> None:
        """Test that get_layer_activation errors on an out-of-bounds layer."""
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        with self.assertRaises(IndexError):
            group_info.get_layer_activation(-5)
        with self.assertRaises(IndexError):
            group_info.get_layer_activation(13)

    # set layer activation works on output layer
    def test_group_info__set_activation_output(self) -> None:
        """Test that set_layer_activation works on the output layer."""
        labels = Labels("a", "b", "c")
        features = ["alpha", "beta", "gamma"]
        group_info = SimpleGroupInfo(features)
        group_info.labels = labels
        group_info.set_output_activation("aleph")
        self.assertEqual("aleph", group_info.get_layer_activation(1))
