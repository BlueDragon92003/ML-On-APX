"""Tests for the GroupInfo class."""

import unittest

from ml_on_apx.labelling import Label, Labels
from ml_on_apx.model_management.group_info import GroupInfo


class TestsGroupInfo(unittest.TestCase):
    """Tests for the GroupInfo class."""

    # All features same as input
    # labels same as input
    # initial features empty
    # layers length starts at 2
    def test_group_info__initialization(self) -> None:
        """Test that the GroupInfo object is insatiated correctly."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        self.assertEqual(labels, group_info.labels)
        self.assertEqual(features, group_info.all_features)
        self.assertEqual(0, len(group_info.features))
        self.assertEqual(0, group_info.layer_count)

    # Enable feaure works
    def test_group_info__enable_feature(self) -> None:
        """Test that enabling a feature works correctly."""
        feature = "alpha"
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {feature, "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.enable_feature(feature)
        self.assertEqual(1, len(group_info.features))
        self.assertIn(feature, group_info.features)

    # disable feature works
    def test_group_info__disable_feature(self) -> None:
        """Test that disabling a feature works correctly."""
        feature = "alpha"
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {feature, "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.enable_feature(feature)
        group_info.disable_feature(feature)
        self.assertEqual(0, len(group_info.features))
        self.assertNotIn(feature, group_info.features)

    # enable unavailable feature errors
    def test_group_info__enable_missing_feature(self) -> None:
        """Test that enabling a non-existant feature errors."""
        feature = "jeff"
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(ValueError):
            group_info.enable_feature(feature)

    # disable unavailable feature errors
    def test_group_info__disable_missing_feature(self) -> None:
        """Test that disabling a non-existant feature errors."""
        feature = "jeff"
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(ValueError):
            group_info.disable_feature(feature)

    # enable enabled feature silently "fails"
    def test_group_info__enable_enabled_feature(self) -> None:
        """Test that enabling an already-enabled feature does nothing."""
        feature = "alpha"
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {feature, "beta", "gamma"}
        group_info = GroupInfo(labels, features)
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
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {feature, "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.enable_feature(feature)
        group_info.disable_feature(feature)
        self.assertEqual(0, len(group_info.features))
        self.assertNotIn(feature, group_info.features)
        group_info.disable_feature(feature)
        self.assertEqual(0, len(group_info.features))
        self.assertNotIn(feature, group_info.features)

    # add layer above works
    def test_group_info__insert_layer_above(self) -> None:
        """Test that insert_layer_above adds a layer above the target."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.insert_layer_above(1, "aleph", 13)
        self.assertEqual(3, group_info.layer_count)
        self.assertEqual("aleph", group_info.get_layer_activation(1))
        self.assertEqual(13, group_info.get_layer_size(1))
        group_info.insert_layer_above(1, "bet", 7)
        self.assertEqual(4, group_info.layer_count)
        self.assertEqual("aleph", group_info.get_layer_activation(2))
        self.assertEqual(13, group_info.get_layer_size(2))
        self.assertEqual("bet", group_info.get_layer_activation(1))
        self.assertEqual(7, group_info.get_layer_size(1))

    # add layer above errors when targeting input layer
    def test_group_info__insert_layer_above_input(self) -> None:
        """Test that insert_layer_above errors when given the input layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.insert_layer_above(0, "aleph", 13)

    # add layer above errors when targeting oob
    def test_group_info__insert_layer_above_oob(self) -> None:
        """Test that insert_layer_above errors when given an out-of-bounds layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.insert_layer_above(13, "aleph", 13)
        with self.assertRaises(IndexError):
            group_info.insert_layer_above(-1, "aleph", 13)

    # add layer below works
    def test_group_info__insert_layer_below(self) -> None:
        """Test that insert_layer_below adds a layer below the target."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.insert_layer_below(0, "aleph", 13)
        self.assertEqual(3, group_info.layer_count)
        self.assertEqual("aleph", group_info.get_layer_activation(1))
        group_info.insert_layer_below(0, "bet", 7)
        self.assertEqual(4, group_info.layer_count)
        self.assertEqual("aleph", group_info.get_layer_activation(2))
        self.assertEqual("bet", group_info.get_layer_activation(1))

    # add layer below erros when targeting output layer
    def test_group_info__insert_layer_below_output(self) -> None:
        """Test that insert_layer_below erros when given the output layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.insert_layer_below(1, "aleph", 13)

    # add layer above errors when targeting oob
    def test_group_info__insert_layer_below_oob(self) -> None:
        """Test that insert_layer_below errors when given an out-of-bounds layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.insert_layer_below(13, "aleph", 13)
        with self.assertRaises(IndexError):
            group_info.insert_layer_below(-1, "aleph", 13)

    # remove layer works
    def test_group_info__remove_layer(self) -> None:
        """Test that remove_layer removes a layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.insert_layer_above(1, "aleph", 13)
        group_info.remove_layer(1)
        self.assertEqual(2, group_info.layer_count)

    # remove layer errors when targeting input layer
    def test_group_info__remove_input(self) -> None:
        """Test that remove_layer errors when given the input layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(ValueError):
            group_info.remove_layer(0)

    # remove layer errors when targeting output layer
    def test_group_info__remove_output(self) -> None:
        """Test that remove_layer errors when given the output layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(ValueError):
            group_info.remove_layer(1)

    # remove layer errors when targeting oob layer
    def test_group_info__remove_out_of_bounds(self) -> None:
        """Test that remove_layer errors when given an out-of-bounds layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.remove_layer(-5)
        with self.assertRaises(IndexError):
            group_info.remove_layer(13)

    # get layer size works for input layer
    def test_group_info__get_size_input(self) -> None:
        """Test that get_layer_size functions on the input layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        self.assertEqual(len(group_info.features), group_info.get_layer_size(0))
        group_info.enable_feature("alpha")
        self.assertEqual(len(group_info.features), group_info.get_layer_size(0))
        group_info.enable_feature("beta")
        self.assertEqual(len(group_info.features), group_info.get_layer_size(0))

    # get layer size works for output layer
    def test_group_info__get_size_output(self) -> None:
        """Test that get_layer_size functions on the output layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        self.assertEqual(len(labels), group_info.get_layer_size(1))

    # get layer size works for hidden layers
    def test_group_info__get_size_hidden(self) -> None:
        """Test that get_layer_size functions on a hidden layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.insert_layer_above(1, "aleph", 13)
        self.assertEqual(13, group_info.get_layer_size(1))

    # get layer size works for oob layers
    def test_group_info__get_size_out_of_bounds(self) -> None:
        """Test that get_layer_size functions on an out-of-bounds layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.get_layer_size(13)
        with self.assertRaises(IndexError):
            group_info.get_layer_size(-5)

    # set layer size works
    def test_group_info__set_size(self) -> None:
        """Test that set_layer_size works on a hidden layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.insert_layer_above(1, "aleph", 13)
        group_info.set_layer_size(1, 7)
        self.assertEqual(7, group_info.get_layer_size(1))

    # set layer size errors on a too-small size
    def test_group_info__set_size_too_small(self) -> None:
        """Test that set_layer_size errors on a non-positive size."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.insert_layer_above(1, "aleph", 13)
        with self.assertRaises(ValueError):
            group_info.set_layer_size(1, -5)

    # set layer size errors on input layer
    def test_group_info__set_size_input(self) -> None:
        """Test that set_layer_size errors on the input layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.set_layer_size(0, 7)

    # set layer size errors on output layer
    def test_group_info__set_size_output(self) -> None:
        """Test that set_layer_size errors on the output layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.set_layer_size(1, 7)

    # set layer size works for oob layers
    def test_group_info__set_size_out_of_bounds(self) -> None:
        """Test that set_layer_size functions on an out-of-bounds layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.set_layer_size(13, 7)
        with self.assertRaises(IndexError):
            group_info.set_layer_size(-5, 7)

    # change layer size works
    def test_group_info__change_size(self) -> None:
        """Test that change_layer_size works on a hidden layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.insert_layer_above(1, "aleph", 13)
        group_info.change_layer_size(1, -6)
        self.assertEqual(7, group_info.get_layer_size(1))
        group_info.change_layer_size(1, 6)
        self.assertEqual(13, group_info.get_layer_size(1))
        group_info.change_layer_size(1, 0)
        self.assertEqual(13, group_info.get_layer_size(1))

    # change layer size errors on input layer
    def test_group_info__change_size_too_small(self) -> None:
        """Test that change_layer_size errors on a too-big negative change."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.insert_layer_above(1, "aleph", 13)
        with self.assertRaises(ValueError):
            group_info.change_layer_size(1, -40)

    # change layer size errors on input layer
    def test_group_info__change_size_input(self) -> None:
        """Test that change_layer_size errors on the input layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.change_layer_size(0, 7)

    # change layer size errors on output layer
    def test_group_info__change_size_output(self) -> None:
        """Test that change_layer_size errors on the output layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.change_layer_size(1, 7)

    # change layer size errors on oob layer
    def test_group_info__change_size_oob(self) -> None:
        """Test that change_layer_size errors on an out-of-bounds layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.change_layer_size(13, 7)
        with self.assertRaises(IndexError):
            group_info.change_layer_size(-5, 7)

    # get layer activation works on output layer
    def test_group_info__get_activation_output(self) -> None:
        """Test that get_layer_activation works on the output layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        self.assertEqual(
            GroupInfo.DEFAULT_ACTIVATION, group_info.get_layer_activation(1)
        )

    # get layer activation errors on input layer
    def test_group_info__get_activation_input(self) -> None:
        """Test that get_layer_activation errors on the input layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(ValueError):
            group_info.get_layer_activation(0)

    # get layer activation works on hidden layer
    def test_group_info__get_activation_hidden(self) -> None:
        """Test that get_layer_activation works on hidden layers."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.insert_layer_above(1, "aleph", 13)
        self.assertEqual("aleph", group_info.get_layer_activation(1))

    # get layer activation errors on oob layer
    def test_group_info__get_activation_oob(self) -> None:
        """Test that get_layer_activation errors on an out-of-bounds layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.get_layer_activation(-5)
        with self.assertRaises(IndexError):
            group_info.get_layer_activation(13)

    # set layer activation works on output layer
    def test_group_info__set_activation_output(self) -> None:
        """Test that set_layer_activation works on the output layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.set_layer_activation(1, "aleph")
        self.assertEqual("aleph", group_info.get_layer_activation(1))

    # set layer activation errors on input layer
    def test_group_info__set_activation_input(self) -> None:
        """Test that set_layer_activation works on the input layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(ValueError):
            group_info.set_layer_activation(0, "aleph")

    # set layer activation works on hidden layer
    def test_group_info__set_activation_hidden(self) -> None:
        """Test that set_layer_activation works on hidden layers."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        group_info.insert_layer_above(1, "bet", 7)
        group_info.set_layer_activation(1, "aleph")
        self.assertEqual("aleph", group_info.get_layer_activation(1))

    # get layer activation errors on oob layer
    def test_group_info__set_activation_oob(self) -> None:
        """Test that set_layer_activation errors on an out-of-bounds layer."""
        labels = Labels([Label("a"), Label("b"), Label("c")])
        features = {"alpha", "beta", "gamma"}
        group_info = GroupInfo(labels, features)
        with self.assertRaises(IndexError):
            group_info.set_layer_activation(-5, "aleph")
        with self.assertRaises(IndexError):
            group_info.set_layer_activation(13, "aleph")
