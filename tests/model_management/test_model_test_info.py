"""Tests for the ModelTestInfo class."""

import datetime
import unittest

from ml_on_apx.model_management.model_info import ModelTestInfo


class TestsModelTestInfo(unittest.TestCase):
    """Tests for the ModelTestInfo class."""

    def test_model_test_info(self) -> None:
        """Test that a ModelTestInfo object instantiates correctly."""
        time = datetime.datetime(2026, 7, 17, 8, 15, 00)
        model_test_info = ModelTestInfo(
            time,
            0.95,
            0.01,
        )
        self.assertEqual(time, model_test_info.test_time)
        self.assertEqual(0.95, model_test_info.accuracy)
        self.assertEqual(0.01, model_test_info.average_loss)
        self.assertFalse(model_test_info.run_by_user)

    def test_model_test_info_run_by_user(self) -> None:
        """Test that a ModelTestInfo object instantiates correctly."""
        time = datetime.datetime(2026, 7, 17, 8, 15, 00)
        model_test_info = ModelTestInfo(time, 0.95, 0.01, run_by_user=True)
        self.assertEqual(time, model_test_info.test_time)
        self.assertEqual(0.95, model_test_info.accuracy)
        self.assertEqual(0.01, model_test_info.average_loss)
        self.assertTrue(model_test_info.run_by_user)
