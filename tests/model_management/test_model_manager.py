"""Tests for the model manager class."""

import datetime
import pickle
import unittest
from pathlib import Path
from typing import ClassVar

import pyfakefs.fake_filesystem_unittest
import torch
from torch import nn

from ml_on_apx.labelling import Label, Labels
from ml_on_apx.model_management.group_info import GroupInfo
from ml_on_apx.model_management.model_info import ModelInfo
from ml_on_apx.model_management.model_manager import ModelManager
from ml_on_apx.model_management.stop_functions import StopFunction
from ml_on_apx.model_management.testing_job import TestingJob
from ml_on_apx.model_management.training_job import TrainingJob
from ml_on_apx.modes import Mode


class TODOError(Exception):
    """Stuff I still need to do."""


class TestsModelManager(
    unittest.TestCase, pyfakefs.fake_filesystem_unittest.TestCaseMixin
):
    """Tests for the ModelManager class."""

    DEFAULT_LABELS: ClassVar[Labels] = Labels([Label("a"), Label("b"), Label("c")])
    DEFAULT_FEATURES: ClassVar[set[str]] = {
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
    }

    def setUp(self) -> None:
        """Set up the filesystem for each test case."""
        self.setUpPyfakefs()
        self.models_path = Path("models")
        self.mode_path = self.models_path / Mode.Testing.value
        self.jobs_pickle = self.mode_path / ModelManager.JOBS_FILE
        self.mode_path.mkdir(parents=True)

    def _create_fs_group(self, group_name: str, group_info: GroupInfo) -> None:
        path = self.mode_path / group_name
        path.mkdir()
        with open(path / ModelManager.GROUP_INFO_FILE, mode="wb") as file:
            pickle.dump(group_info, file)
        with open(path / ModelManager.MODEL_INFOS_FILE, mode="wb") as file:
            pickle.dump({}, file)

    def _create_fs_model(
        self, group_name: str, model_name: str, model_info: ModelInfo
    ) -> None:
        group_path = self.mode_path / group_name
        if not group_path.exists():
            self._create_fs_group(
                group_name, GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
        model_infos_path = group_path / ModelManager.MODEL_INFOS_FILE
        with open(model_infos_path, mode="rb") as file:
            model_infos: dict[str, ModelInfo] = pickle.load(file)
        model_infos.update({model_name: model_info})
        with open(model_infos_path, mode="wb") as file:
            pickle.dump(model_infos, file)
        model_path = group_path / (model_name + ModelManager.PYTORCH_SUFFIX)
        with open(model_path, mode="wb") as file:
            torch.save(nn.Module(), file)

    def _set_training_job(self, training_job: TrainingJob | None) -> None:
        jobs = (training_job, None)
        if self.jobs_pickle.exists():
            with open(self.jobs_pickle, mode="rb") as file:
                cur_jobs: tuple[TrainingJob | None, TestingJob | None] = pickle.load(
                    file
                )
                jobs = (training_job, cur_jobs[1])
        with open(self.jobs_pickle, mode="wb") as file:
            pickle.dump(jobs, file)

    def _set_testing_job(self, testing_job: TestingJob | None) -> None:
        jobs = (None, testing_job)
        if self.jobs_pickle.exists():
            with open(self.jobs_pickle, mode="rb") as file:
                cur_jobs: tuple[TrainingJob | None, TestingJob | None] = pickle.load(
                    file
                )
                jobs = (cur_jobs[0], testing_job)
        with open(self.jobs_pickle, mode="wb") as file:
            pickle.dump(jobs, file)

    # ========================================================================
    #                                 ENTERING
    # ========================================================================

    # Without jobs file (Lists are None)
    def test_model_manager__enter__no_jobs_file(self) -> None:
        """Test that the model manager does not error if the jobs file is missing."""
        with ModelManager(self.models_path, Mode.Testing) as _:
            pass

    # With jobs file (lists appear as expected)
    def test_model_manager__enter__r_jobs_file(self) -> None:
        """Test that the model manager successfully loads a training job."""
        job = TrainingJob.new()
        job.group_name("grp")
        job.dataset("data")
        job.stop_function(StopFunction("(quote nil)"))
        job = job.build()
        self._set_training_job(job)
        with ModelManager(self.models_path, Mode.Testing) as manager:
            saved = manager.training_job
            assert saved is not None  # Only here because my typechecker isn't too smart
            self.assertEqual(job._group_name, saved._group_name)
            self.assertEqual(job._dataset, saved._dataset)
            self.assertEqual(job._stop_function._code, saved._stop_function._code)
            self.assertEqual(job._lookback_distance, saved._lookback_distance)
            self.assertEqual(job._batch_size, saved._batch_size)
            self.assertEqual(job._checkpoint_rate, saved._checkpoint_rate)
            self.assertEqual(job._learning_rate, saved._learning_rate)
            self.assertEqual(job._testing_dataset, saved._testing_dataset)
            self.assertEqual(job._base_model_name, saved._base_model_name)

    def test_model_manager__enter__e_jobs_file(self) -> None:
        """Test that the model manager successfully loads a testing job."""
        job = TestingJob(Path("/ignore"), "dataset")
        self._set_testing_job(job)
        with ModelManager(self.models_path, Mode.Testing) as manager:
            testing = manager.testing_job
            assert testing is not None
            self.assertEqual(job._target, testing._target)
            self.assertEqual(job._dataset, testing._dataset)

    def test_model_manager__enter__re_jobs_file(self) -> None:
        """Test that the model manager successfully loads both jobs."""
        training_job = TrainingJob.new()
        training_job.group_name("grp")
        training_job.dataset("data")
        training_job.stop_function(StopFunction("(quote nil)"))
        training_job = training_job.build()
        testing_job = TestingJob(Path("/ignore"), "dataset")
        self._set_training_job(training_job)
        self._set_testing_job(testing_job)
        with ModelManager(self.models_path, Mode.Testing) as manager:
            saved_training = manager.training_job
            saved_testing = manager.testing_job
            assert saved_training is not None
            assert saved_testing is not None
            self.assertEqual(training_job._group_name, saved_training._group_name)
            self.assertEqual(training_job._dataset, saved_training._dataset)
            self.assertEqual(
                training_job._stop_function._code, saved_training._stop_function._code
            )
            self.assertEqual(
                training_job._lookback_distance, saved_training._lookback_distance
            )
            self.assertEqual(training_job._batch_size, saved_training._batch_size)
            self.assertEqual(
                training_job._checkpoint_rate, saved_training._checkpoint_rate
            )
            self.assertEqual(training_job._learning_rate, saved_training._learning_rate)
            self.assertEqual(
                training_job._testing_dataset, saved_training._testing_dataset
            )
            self.assertEqual(
                training_job._base_model_name, saved_training._base_model_name
            )
            self.assertEqual(testing_job._target, saved_testing._target)
            self.assertEqual(testing_job._dataset, saved_testing._dataset)

    # Filters out files in group dir (`group_names`)
    def test_model_manager__enter__group_folders_only(self) -> None:
        """Test that the model manager loads only directories, not files, as groups."""
        valid_group_names = ["group_a", "group_c", "group_d", "group_g"]
        for group_name in valid_group_names:
            self._create_fs_group(
                group_name, GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
        for group_name in ["group_b", "group_e", "group_f"]:
            self._create_fs_group(
                group_name, GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
            path = self.mode_path / group_name
            for file in path.iterdir():
                file.unlink()
            path.rmdir()
            path.touch()
        with ModelManager(self.models_path, Mode.Testing) as manager:
            self.assertEqual(valid_group_names, manager.group_names)

    # Filters out missing group info
    def test_model_manager__enter__filter_missing_group_info(self) -> None:
        """Test that the model manager skips groups missing `group_info.pckl`."""
        valid_group_names = ["group_a", "group_c", "group_e", "group_g"]
        for group_name in valid_group_names:
            self._create_fs_group(
                group_name, GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
        for group_name in ["group_b", "group_d", "group_f"]:
            self._create_fs_group(
                group_name, GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
            path = self.mode_path / group_name / ModelManager.GROUP_INFO_FILE
            path.unlink()
        with ModelManager(self.models_path, Mode.Testing) as manager:
            self.assertEqual(valid_group_names, manager.group_names)

    # Filters out missing model info
    def test_model_manager__enter__filter_missing_model_infos(self) -> None:
        """Test that the model manager skips groups missing `model_infos`.pckl."""
        valid_group_names = ["group_b", "group_c", "group_d", "group_g"]
        for group_name in valid_group_names:
            self._create_fs_group(
                group_name, GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
        for group_name in ["group_a", "group_e", "group_f"]:
            self._create_fs_group(
                group_name, GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
            path = self.mode_path / group_name / ModelManager.MODEL_INFOS_FILE
            path.unlink()
        with ModelManager(self.models_path, Mode.Testing) as manager:
            self.assertEqual(valid_group_names, manager.group_names)

    # Filters out `model_info`s missing `model`s
    def test_model_manager__enter__filter_missing_models(self) -> None:
        """Test that the model manager filters out model infos without `.pth` models."""
        group = "grp"
        valid_model_names = ["model_1", "model_3", "model_4", "model_7"]
        for name in valid_model_names:
            self._create_fs_model(
                group,
                name,
                ModelInfo(
                    datetime.date.today(), datetime.datetime.now(), group, "data"
                ),
            )
        for name in ["model_2", "model_5", "model_6", "model_8"]:
            self._create_fs_model(
                group,
                name,
                ModelInfo(
                    datetime.date.today(), datetime.datetime.now(), group, "data"
                ),
            )
            (self.mode_path / group / (name + ModelManager.PYTORCH_SUFFIX)).unlink()
        with ModelManager(self.models_path, Mode.Testing) as manager:
            self.assertEqual(valid_model_names, manager.get_model_names(group))

    # ========================================================================
    #                               CREATE GROUP
    # ========================================================================

    # Name already exists
    def test_model_manager__cg__name_exists(self) -> None:
        """Test group creation with a pre-existing name."""
        self._create_fs_group(
            "group_a", GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
        )
        with ModelManager(self.models_path, Mode.Testing) as manager:
            with self.assertRaises(ValueError):
                manager.create_group(
                    "group_a", GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
                )

    # OK
    def test_model_manager__cg__okay(self) -> None:
        """Test functioning group creation."""
        self._create_fs_group(
            "group_a", GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
        )
        with ModelManager(self.models_path, Mode.Testing) as manager:
            manager.create_group(
                "group_b", GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
            self.assertTrue((self.mode_path / "group_b").exists())

    # ========================================================================
    #                                READ GROUP
    # ========================================================================

    # No such group
    def test_model_manager__rg__missing_group(self) -> None:
        """Test that read group errors if given a nonexistent group."""
        with ModelManager(self.models_path, Mode.Testing) as manager:
            with self.assertRaises(ModelManager.GroupLookupError):
                manager.get_group_info("group_a")

    # OK
    def test_model_manager__rg__ok(self) -> None:
        """Test that read group behaves as expeced in normal circumstances."""
        label_2 = Labels([Label("a"), Label("b")])
        feature_2 = {"omega", "sigma"}
        self._create_fs_group("group_a", GroupInfo(self.DEFAULT_LABELS, feature_2))
        with ModelManager(self.models_path, Mode.Testing) as manager:
            manager.create_group("group_b", GroupInfo(label_2, self.DEFAULT_FEATURES))
            info = manager.get_group_info("group_a")
            self.assertEqual(self.DEFAULT_LABELS, info.labels)
            self.assertEqual(feature_2, info.all_features)
            info = manager.get_group_info("group_b")
            self.assertEqual(label_2, info.labels)
            self.assertEqual(self.DEFAULT_FEATURES, info.all_features)

    # ========================================================================
    #                               UPDATE GROUP
    # ========================================================================

    # Group DNE
    def test_model_manager__urg__missing_group(self) -> None:
        """Test that rename group errors if given a nonexistent source group."""
        with ModelManager(self.models_path, Mode.Testing) as manager:
            with self.assertRaises(ModelManager.GroupLookupError):
                manager.rename_group("group_a", "group_b")

    # Name already used
    def test_model_manager__urg__used_name(self) -> None:
        """Test that rename group errors if given a extant target group."""
        with ModelManager(self.models_path, Mode.Testing) as manager:
            manager.create_group(
                "group_a", GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
            manager.create_group(
                "group_b", GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
            with self.assertRaises(ValueError):
                manager.rename_group("group_a", "group_b")

    # OK
    def test_model_manager__urg__ok(self) -> None:
        """Test that rename group behaves as expeced."""
        with ModelManager(self.models_path, Mode.Testing) as manager:
            manager.create_group(
                "group_a", GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
            manager.rename_group("group_a", "group_b")
            with self.assertRaises(ModelManager.GroupLookupError):
                manager.get_group_info("group_a")
            info = manager.get_group_info("group_b")
            self.assertEqual(self.DEFAULT_LABELS, info.labels)
            self.assertEqual(self.DEFAULT_FEATURES, info.all_features)

    # ========================================================================
    #                               DELETE GROUP
    # ========================================================================

    # Group DNE
    def test_model_manager__dg__missing_group(self) -> None:
        """Test that delete group errors if given a nonexistent group."""
        with ModelManager(self.models_path, Mode.Testing) as manager:
            with self.assertRaises(ModelManager.GroupLookupError):
                manager.delete_group("group_a")

    # OK
    def test_model_manager__dg__ok(self) -> None:
        """Test that delete group behaves as expected."""
        with ModelManager(self.models_path, Mode.Testing) as manager:
            manager.create_group(
                "group_a", GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
            manager.delete_group("group_a")
            with self.assertRaises(ModelManager.GroupLookupError):
                manager.get_group_info("group_a")

    # ========================================================================
    #                               CREATE MODEL
    # ========================================================================

    # Group DNE
    def test_model_manager__cm__missing_group(self) -> None:
        """Test that create model errors if given a nonexistent group."""
        raise TODOError  # TODO

    # Name already used
    def test_model_manager__cm__name_used(self) -> None:
        """Test that create model errors the model name is already in use."""
        raise TODOError  # TODO

    # OK
    def test_model_manager__cm__ok(self) -> None:
        """Test that create model behaves as expected."""
        raise TODOError  # TODO

    # ========================================================================
    #                                READ MODEL
    # ========================================================================

    # Info::Group DNE
    def test_model_manager__rmi__missing_group(self) -> None:
        """Test that read model info errors if given a nonexistent group."""
        raise TODOError  # TODO

    # Info::Model DNE
    def test_model_manager__rmi__missing_model(self) -> None:
        """Test that read model info errors if given a nonexistent model."""
        raise TODOError  # TODO

    # Info::OK
    def test_model_manager__rmi__ok(self) -> None:
        """Test that read model info behaves as expected."""
        raise TODOError  # TODO

    # Model::Group DNE
    def test_model_manager__rm__missing_group(self) -> None:
        """Test that read model errors if given a nonexistent group."""
        raise TODOError  # TODO

    # Model::Model DNE
    def test_model_manager__rm__missing_model(self) -> None:
        """Test that read model errors if given a nonexistent model."""
        raise TODOError  # TODO

    # Model::OK
    def test_model_manager__rm__ok(self) -> None:
        """Test that read model behaves as expected."""
        raise TODOError  # TODO

    # ========================================================================
    #                               UPDATE MODEL
    # ========================================================================

    # Rename::Group DNE
    def test_model_manager__urm__missing_group(self) -> None:
        """Test that rename model errors if given a nonexistent group."""
        raise TODOError  # TODO

    # Rename::Model DNE
    def test_model_manager__urm__missing_model(self) -> None:
        """Test that rename model errors if given a nonexistent model."""
        raise TODOError  # TODO

    # Rename::Name already used
    def test_model_manager__urm__name_used(self) -> None:
        """Test that rename model errors if the name is already in use."""
        raise TODOError  # TODO

    # Rename::OK
    def test_model_manager__urm__ok(self) -> None:
        """Test that rename model behaves as expected."""
        raise TODOError  # TODO

    # Update::Group DNE
    def test_model_manager__um__missing_group(self) -> None:
        """Test that update model errors if given a nonexistent group."""
        raise TODOError  # TODO

    # Update::Model DNE
    def test_model_manager__um__missing_model(self) -> None:
        """Test that update model errors if given a nonexistent model."""
        raise TODOError  # TODO

    # Update::OK
    def test_model_manager__um__ok(self) -> None:
        """Test that update model behaves as expected."""
        raise TODOError  # TODO

    # ========================================================================
    #                               DELETE MODEL
    # ========================================================================

    # Group DNE
    def test_model_manager__dm__missing_group(self) -> None:
        """Test that delete model errors if given a nonexistent group."""
        raise TODOError  # TODO

    # Model DNE
    def test_model_manager__dm__missing_model(self) -> None:
        """Test that delete model errors if given a nonexistent model."""
        raise TODOError  # TODO

    # OK
    def test_model_manager__dm__ok(self) -> None:
        """Test that delete model behaves as expected."""
        raise TODOError  # TODO

    # ========================================================================
    #                                   JOBS
    # ========================================================================

    # Training Job setter works
    def test_model_manager__job_setter__training(self) -> None:
        """Test that the setter for the training job works."""
        job = TrainingJob.new()
        job.group_name("grp")
        job.dataset("data")
        job.stop_function(StopFunction("(quote nil)"))
        job = job.build()
        with ModelManager(self.models_path, Mode.Testing) as manager:
            self.assertIsNone(manager.training_job)
            manager.training_job = job
            set = manager.training_job
            assert set is not None  # Only here because my typechecker isn't too smart
            self.assertEqual(job._group_name, set._group_name)
            self.assertEqual(job._dataset, set._dataset)
            self.assertEqual(job._stop_function._code, set._stop_function._code)
            self.assertEqual(job._lookback_distance, set._lookback_distance)
            self.assertEqual(job._batch_size, set._batch_size)
            self.assertEqual(job._checkpoint_rate, set._checkpoint_rate)
            self.assertEqual(job._learning_rate, set._learning_rate)
            self.assertEqual(job._testing_dataset, set._testing_dataset)
            self.assertEqual(job._base_model_name, set._base_model_name)

    # Testing Job setter works
    def test_model_manager__job_setter__testing(self) -> None:
        """Test that the setter for the testing job works."""
        job = TestingJob(Path("/ignore"), "dataset")
        with ModelManager(self.models_path, Mode.Testing) as manager:
            self.assertIsNone(manager.testing_job)
            manager.testing_job = job
            testing = manager.testing_job
            assert testing is not None
            self.assertEqual(job._target, testing._target)
            self.assertEqual(job._dataset, testing._dataset)
