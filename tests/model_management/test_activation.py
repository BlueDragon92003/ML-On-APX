"""Tests for the Activation class."""

import unittest

import torch

from ml_on_apx.model_management.group_info import Activation


class TestsActivation(unittest.TestCase):
    """Tests for the Activation class."""

    def test_activation__instantiation(self) -> None:
        """Test that making a new activation is painless."""
        activation = Activation("sigmoid", torch.nn.Sigmoid)
        self.assertEqual("sigmoid", activation.name)
        self.assertIs(torch.nn.Sigmoid, activation.activation)

    def test_activation__equality(self) -> None:
        """Test that two equal activations are equal."""
        activation1 = Activation("sigmoid", torch.nn.Sigmoid)
        activation2 = Activation("jeff", torch.nn.Sigmoid)
        self.assertEqual(activation1, activation2)

    def test_activation__inequality(self) -> None:
        """Test that two not equal activations are not equal."""
        activation1 = Activation("jeff", torch.nn.Sigmoid)
        activation2 = Activation("jeff", torch.nn.Softplus)
        self.assertNotEqual(activation1, activation2)
