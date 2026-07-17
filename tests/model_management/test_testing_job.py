"""Tests for the TestingJob class."""

import unittest
from pathlib import Path

from ml_on_apx.model_management.testing_job import TestingJob


class TestsTestingJob(unittest.TestCase):
    """Tests for the TestingJob class."""

    def test_testing_job(self) -> None:
        """Test that a TestingJob is created correctly."""
        target = Path("/home/user/app/models/group/model.pth")
        dataset = "dataset_1"
        job = TestingJob(target, dataset)
        self.assertEqual(target, job.target)
        self.assertEqual(dataset, job.dataset)
