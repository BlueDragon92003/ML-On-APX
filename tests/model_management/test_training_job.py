"""Tests for the TrainingJob class."""

import unittest

from eliot.testing import capture_logging

from ml_on_apx.model_management.stop_functions import StopFunction
from ml_on_apx.model_management.training_job import TrainingJob


class TestsTrainingJob(unittest.TestCase):
    """Tests for the TrainingJob class."""

    # minimal functions
    @capture_logging
    def test_training_job__minimal_build(self) -> None:
        """Test that a minimal valid build builds properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)
        job = builder.build()
        self.assertEqual(group_name, job.group_name)
        self.assertEqual(dataset, job.dataset)
        self.assertEqual(3, job.lookback_distance)
        self.assertEqual(1, job.batch_size)
        self.assertEqual(10, job.checkpoint_rate)
        self.assertAlmostEqual(1e-4, job.learning_rate)
        self.assertEqual(None, job.testing_dataset)
        self.assertEqual(None, job.base_model_name)

    # missing group name
    @capture_logging
    def test_training_job__missing_group(self) -> None:
        """Test that a build missing its group does not build properly."""
        # group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        builder = TrainingJob.new()
        # builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)
        with self.assertRaises(TypeError):
            builder.build()

    # missing dataset
    @capture_logging
    def test_training_job__missing_dataset(self) -> None:
        """Test that a build missing its dataset does not build properly."""
        group_name = "grp"
        # dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        builder = TrainingJob.new()
        builder.group_name(group_name)
        # builder.dataset(dataset)
        builder.stop_function(stop_func)
        with self.assertRaises(TypeError):
            builder.build()

    # missing stop function
    @capture_logging
    def test_training_job__missing_stop(self) -> None:
        """Test that a build missing its stop_function does not build properly."""
        group_name = "grp"
        dataset = "train"
        # stop_func = StopFunction(
        #     "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        # )
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        # builder.stop_function(stop_func)
        with self.assertRaises(TypeError):
            builder.build()

    # functions + lookback
    @capture_logging
    def test_training_job__lookback_build(self) -> None:
        """Test that a build with a set lookback distance builds properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        lookback = 15
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)
        builder.lookback_distance(lookback)
        job = builder.build()
        self.assertEqual(group_name, job.group_name)
        self.assertEqual(dataset, job.dataset)
        self.assertEqual(lookback, job.lookback_distance)
        self.assertEqual(1, job.batch_size)
        self.assertEqual(10, job.checkpoint_rate)
        self.assertAlmostEqual(1e-4, job.learning_rate)
        self.assertEqual(None, job.testing_dataset)
        self.assertEqual(None, job.base_model_name)

    # functions + batch size
    @capture_logging
    def test_training_job__batch_size_build(self) -> None:
        """Test that a build with a set batch size builds properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        batch_size = 5
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)
        builder.batch_size(batch_size)
        job = builder.build()
        self.assertEqual(group_name, job.group_name)
        self.assertEqual(dataset, job.dataset)
        self.assertEqual(3, job.lookback_distance)
        self.assertEqual(batch_size, job.batch_size)
        self.assertEqual(10, job.checkpoint_rate)
        self.assertAlmostEqual(1e-4, job.learning_rate)
        self.assertEqual(None, job.testing_dataset)
        self.assertEqual(None, job.base_model_name)

    # functions + checkpoint
    @capture_logging
    def test_training_job__checkpoint_build(self) -> None:
        """Test that a build with a set checkpoint rate builds properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        checkpoint_rate = 5
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)
        builder.checkpoint_rate(checkpoint_rate)
        job = builder.build()
        self.assertEqual(group_name, job.group_name)
        self.assertEqual(dataset, job.dataset)
        self.assertEqual(3, job.lookback_distance)
        self.assertEqual(1, job.batch_size)
        self.assertEqual(checkpoint_rate, job.checkpoint_rate)
        self.assertAlmostEqual(1e-4, job.learning_rate)
        self.assertEqual(None, job.testing_dataset)
        self.assertEqual(None, job.base_model_name)

    # functions + learning rate
    @capture_logging
    def test_training_job__learning_build(self) -> None:
        """Test that a build with a set learning rate builds properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        learning_rate = 7.62e-3
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)
        builder.learning_rate(learning_rate)
        job = builder.build()
        self.assertEqual(group_name, job.group_name)
        self.assertEqual(dataset, job.dataset)
        self.assertEqual(3, job.lookback_distance)
        self.assertEqual(1, job.batch_size)
        self.assertEqual(10, job.checkpoint_rate)
        self.assertAlmostEqual(learning_rate, job.learning_rate)
        self.assertEqual(None, job.testing_dataset)
        self.assertEqual(None, job.base_model_name)

    # invalid lookback
    @capture_logging
    def test_training_job__invalid_lookback(self) -> None:
        """Test that a build with an invalid lookback does not build properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)

        builder.lookback_distance(0)
        with self.assertRaises(ValueError):
            builder.build()
        builder.lookback_distance(-5)
        with self.assertRaises(ValueError):
            builder.build()

    # invalid batch size
    @capture_logging
    def test_training_job__invalid_batch(self) -> None:
        """Test that a build with an invalid batch size does not build properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)

        builder.batch_size(0)
        with self.assertRaises(ValueError):
            builder.build()
        builder.batch_size(-5)
        with self.assertRaises(ValueError):
            builder.build()

    # invalid checkpoint rate
    @capture_logging
    def test_training_job__invalid_checkpoint(self) -> None:
        """Test that a build with an invalid checkpoint rate does not build properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)

        builder.checkpoint_rate(0)
        with self.assertRaises(ValueError):
            builder.build()
        builder.checkpoint_rate(-5)
        with self.assertRaises(ValueError):
            builder.build()

    # invalid learning rate
    @capture_logging
    def test_training_job__invalid_learn(self) -> None:
        """Test that a build with an invalid learning rate does not build properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)

        builder.learning_rate(0)
        with self.assertRaises(ValueError):
            builder.build()
        builder.learning_rate(-5.04)
        with self.assertRaises(ValueError):
            builder.build()

    # functions + testing set
    @capture_logging
    def test_training_job__testing_build(self) -> None:
        """Test that a build with a set testing dataset builds properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        testing_set = "test"
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)
        builder.testing_dataset(testing_set)
        job = builder.build()
        self.assertEqual(group_name, job.group_name)
        self.assertEqual(dataset, job.dataset)
        self.assertEqual(3, job.lookback_distance)
        self.assertEqual(1, job.batch_size)
        self.assertEqual(10, job.checkpoint_rate)
        self.assertAlmostEqual(1e-4, job.learning_rate)
        self.assertEqual(testing_set, job.testing_dataset)
        self.assertEqual(None, job.base_model_name)

    # functions + base model
    @capture_logging
    def test_training_job__base_model_build(self) -> None:
        """Test that a build with a set base model builds properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        base_model = "base"
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)
        builder.base_model_name(base_model)
        job = builder.build()
        self.assertEqual(group_name, job.group_name)
        self.assertEqual(dataset, job.dataset)
        self.assertEqual(3, job.lookback_distance)
        self.assertEqual(1, job.batch_size)
        self.assertEqual(10, job.checkpoint_rate)
        self.assertAlmostEqual(1e-4, job.learning_rate)
        self.assertEqual(None, job.testing_dataset)
        self.assertEqual(base_model, job.base_model_name)

    # maximal functions
    @capture_logging
    def test_training_job__maximal_build(self) -> None:
        """Test that a build with all properties set builds properly."""
        group_name = "grp"
        dataset = "train"
        stop_func = StopFunction(
            "(or (> (car ACC) (car (cdr ACC)) ) (< (car LOSS) (car (cdr LOSS)) ) )"
        )
        lookback = 15
        batch_size = 5
        checkpoint_rate = 5
        learning_rate = 7.62e-3
        testing_set = "test"
        base_model = "base"
        builder = TrainingJob.new()
        builder.group_name(group_name)
        builder.dataset(dataset)
        builder.stop_function(stop_func)
        builder.lookback_distance(lookback)
        builder.batch_size(batch_size)
        builder.checkpoint_rate(checkpoint_rate)
        builder.learning_rate(learning_rate)
        builder.testing_dataset(testing_set)
        builder.base_model_name(base_model)
        job = builder.build()
        self.assertEqual(group_name, job.group_name)
        self.assertEqual(dataset, job.dataset)
        self.assertEqual(lookback, job.lookback_distance)
        self.assertEqual(batch_size, job.batch_size)
        self.assertEqual(checkpoint_rate, job.checkpoint_rate)
        self.assertAlmostEqual(learning_rate, job.learning_rate)
        self.assertEqual(testing_set, job.testing_dataset)
        self.assertEqual(base_model, job.base_model_name)
