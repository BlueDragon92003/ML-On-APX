"""Tests for the model manager class."""

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

    def _create_fs_model(
        self, group_name: str, model_name: str, model_info: ModelInfo
    ) -> None:
        group_path = self.mode_path / group_name
        if not group_path.exists():
            self._create_fs_group(
                group_name, GroupInfo(self.DEFAULT_LABELS, self.DEFAULT_FEATURES)
            )
        model_infos_path = group_path / ModelManager.MODEL_INFOS_FILE
        with open(model_infos_path, mode="w+b") as file:
            model_infos: dict[str, ModelInfo] = pickle.load(file)
            model_infos.update({model_name: model_info})
            pickle.dump(model_infos, file)
        model_path = group_path / (model_name + ModelManager.PYTORCH_SUFFIX)
        with open(model_path, mode="wb") as file:
            torch.save(nn.Module(), file)

    def _set_training_job(self, training_job: TrainingJob) -> None: ...

    def _set_testing_job(self, testing_job: TestingJob) -> None: ...

    # ========================================================================
    #                                 ENTERING
    # ========================================================================

    # Without jobs file (Lists are None)
    def test_model_manager__enter__no_jobs_file(self) -> None:
        """Test that the model manager does not error if the jobs file is missing."""
        raise TODOError  # TODO

    # With jobs file (lists appear as expected)
    def test_model_manager__enter__r_jobs_file(self) -> None:
        """Test that the model manager successfully loads a training job."""
        raise TODOError  # TODO

    def test_model_manager__enter__e_jobs_file(self) -> None:
        """Test that the model manager successfully loads a testing job."""
        raise TODOError  # TODO

    def test_model_manager__enter__re_jobs_file(self) -> None:
        """Test that the model manager successfully loads both jobs."""
        raise TODOError  # TODO

    # Filters out files in group dir (`group_names`)
    def test_model_manager__enter__group_folders_only(self) -> None:
        """Test that the model manager loads only directories, not files, as groups."""
        raise TODOError  # TODO

    # Filters out missing group info
    def test_model_manager__enter__filter_missing_group_info(self) -> None:
        """Test that the model manager skips groups missing `group_info.pckl`."""
        raise TODOError  # TODO

    # Filters out missing model info
    def test_model_manager__enter__filter_missing_model_infos(self) -> None:
        """Test that the model manager skips groups missing `model_infos`.pckl."""
        raise TODOError  # TODO

    # Filters out `model_info`s missing `model`s
    def test_model_manager__enter__filter_missing_models(self) -> None:
        """Test that the model manager filters out model infos without `.pth` models."""
        raise TODOError  # TODO

    # ========================================================================
    #                               CREATE GROUP
    # ========================================================================

    # Name already exists
    def test_model_manager__cg__name_exists(self) -> None:
        """Test group creation with a pre-existing name."""
        raise TODOError  # TODO

    # OK
    def test_model_manager__cg__okay(self) -> None:
        """Test functioning group creation."""
        raise TODOError  # TODO

    # ========================================================================
    #                                READ GROUP
    # ========================================================================

    # No such group
    def test_model_manager__rg__missing_group(self) -> None:
        """Test that read group errors if given a nonexistent group."""
        raise TODOError  # TODO

    # OK
    def test_model_manager__rg__ok(self) -> None:
        """Test that read group behaves as expeced in normal circumstances."""
        raise TODOError  # TODO

    # Get model names
    def test_model_manager__rg__model_names(self) -> None:
        """Test that the model names in a group can sucessfully be read."""
        raise TODOError  # TODO

    # ========================================================================
    #                               UPDATE GROUP
    # ========================================================================

    # Rename::Group DNE
    def test_model_manager__urg__missing_group(self) -> None:
        """Test that rename group errors if given a nonexistent source group."""
        raise TODOError  # TODO

    # Rename::Name already used
    def test_model_manager__urg__used_name(self) -> None:
        """Test that rename group errors if given a extant target group."""
        raise TODOError  # TODO

    # Rename::OK
    def test_model_manager__urg__ok(self) -> None:
        """Test that rename group behaves as expeced."""
        raise TODOError  # TODO

    # Update::GROUP DNE
    def test_model_manager__ug__missing_group(self) -> None:
        """Test that update group errors if given a nonexistent group."""
        raise TODOError  # TODO

    # Update::OK
    def test_model_manager__ug__okay(self) -> None:
        """Test that update group behaves as expected."""
        raise TODOError  # TODO

    # ========================================================================
    #                               DELETE GROUP
    # ========================================================================

    # Group DNE
    def test_model_manager__dg__missing_group(self) -> None:
        """Test that delete group errors if given a nonexistent group."""
        raise TODOError  # TODO

    # OK
    def test_model_manager__dg__ok(self) -> None:
        """Test that delete group behaves as expected."""
        raise TODOError  # TODO

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
        """Test that read group errors if given a nonexistent group."""
        raise TODOError  # TODO

    # Testing Job setter works
    def test_model_manager__job_setter__testing(self) -> None:
        """Test that read group errors if given a nonexistent group."""
        raise TODOError  # TODO
