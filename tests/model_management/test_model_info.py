"""Tests for the ModelInfo class."""

import datetime
import unittest

from ml_on_apx.model_management.model_info import ModelInfo, ModelTestInfo


class TestsModelInfo(unittest.TestCase):
    """Tests for the ModelInfo class."""

    def test_model_info__instantiation(self) -> None:
        """Test that the ModelInfo object is created properly."""
        start = datetime.date(2026, 7, 17)
        fork = datetime.datetime(2026, 7, 17, 8, 26, 00)
        group = "group_i"
        tds = "dataset_1"
        model_info = ModelInfo(start, fork, group, tds)

        self.assertEqual(start, model_info.training_start_date)
        self.assertEqual(fork, model_info.model_fork_time)
        self.assertEqual(group, model_info.group)
        self.assertEqual(tds, model_info.training_dataset)
        self.assertEqual(0, len(model_info.testing_information))

    def test_model_info__add_test_info(self) -> None:
        """Test that adding a ModelTestInfo object functions properly."""
        model_info = ModelInfo(
            datetime.date(2026, 7, 17),
            datetime.datetime(2026, 7, 17, 8, 26, 00),
            "group_i",
            "dataset_1",
        )
        self.assertEqual(0, len(model_info.testing_information))
        test = ModelTestInfo(
            datetime.datetime(2026, 7, 17, 8, 32, 00),
            0.95,
            0.01,
        )
        model_info.add_testing_information(test)
        self.assertEqual(1, len(model_info.testing_information))
        self.assertIn(test, model_info.testing_information)
