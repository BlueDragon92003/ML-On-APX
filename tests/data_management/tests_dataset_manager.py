import pyfakefs.fake_filesystem_unittest
import unittest


class TestDatasetManager(
    unittest.TestCase, pyfakefs.fake_filesystem_unittest.TestCaseMixin
):
    def setUp():
        raise NotImplementedError

    # ========================================================================
    #                                 HELPERS
    # ========================================================================

    def test_dataset_manager__get_dataset_path(self):
        """Test that the manager errors if the root file directory path is a file."""
        raise NotImplementedError

    def test_dataset_manager__datasets_in_dir(self):
        """Test that the helper method produces the correct results."""
        raise NotImplementedError

    def test_dataset_manager__recompile_dataset(self):
        raise NotImplementedError

    # ========================================================================
    #                              INSTANTIATION
    # ========================================================================

    def test_dataset_manager__initialization(self):
        """Test that there are no errors during initialization"""
        raise NotImplementedError

    # ========================================================================
    #                                 ENTERING
    # ========================================================================

    def test_dataset_manager__enter_missing_root_dir(self):
        """Test that the manager creates a directory for ROOT files."""
        raise NotImplementedError

    def test_dataset_manager__enter_missing_sets_dir(self):
        """Test that the manager creates a directory for datasets files."""
        raise NotImplementedError

    def test_dataset_manager__enter_file_at_root_dir(self):
        """Test that the manager errors if the ROOT file directory path is a file."""
        raise NotImplementedError

    def test_dataset_manager__enter_file_at_sets_dir(self):
        """Test that the manager errors if the dataset file directory path is a file."""
        raise NotImplementedError

    def test_dataset_manager__recompiles_on_exit(self):
        """Test that the manager recompiles the appropriate files upon exiting."""
        raise NotImplementedError

    def test_dataset_manager__enters_and_exits_fine(self):
        """Test that the manager successfully operates in a normal scenario."""
        raise NotImplementedError

    # ========================================================================
    #                                 SOURCES
    # ========================================================================

    def test_dataset_manager__get_sources(self):
        """Test that the manager returns the correct tree of available sources."""
        raise NotImplementedError

    # ========================================================================
    #                              CREATE DATASET
    # ========================================================================

    def test_dataset_manager__create(self):
        """Test that the manager can create a new dataset."""
        raise NotImplementedError

    def test_dataset_manager__create_existing(self):
        """Test that the manager errors when a dataset with that name already exists."""
        raise NotImplementedError

    # ========================================================================
    #                              READ DATASETS
    # ========================================================================

    def test_dataset_manager__names(self):
        """Test that the manager delivers the correct list of available datasets."""
        raise NotImplementedError

    def test_dataset_manager__get_dataset_info(self):
        """Test that the manager delivers the correct dataset_info object."""
        raise NotImplementedError

    def test_dataset_manager__get_dataset_info_missing(self):
        """Test that the manager errors when there is no dataset_info object."""
        raise NotImplementedError

    def test_dataset_manager__get_dataset(self):
        """Test that the manager delivers the correct dataset object."""
        raise NotImplementedError

    def test_dataset_manager__get_dataset_missing(self):
        """Test that the manager errors when there is no dataset object."""
        raise NotImplementedError

    # ========================================================================
    #                             UPDATE DATASETS
    # ========================================================================

    def test_dataset_manager__update(self):
        """Test that the manager correctly updates a dataset's information."""
        raise NotImplementedError

    def test_dataset_manager__update_missing(self):
        """Test that the manager errors when there is no dataset to update."""
        raise NotImplementedError

    def test_dataset_manager__rename(self):
        """Test that the manager correctly renames a dataset."""
        raise NotImplementedError

    def test_dataset_manager__rename_missing(self):
        """Test that the manager errors when there is no dataset to rename."""
        raise NotImplementedError

    def test_dataset_manager__rename_to_existing(self):
        """Test that the manager errors when the target already exists."""
        raise NotImplementedError

    # ========================================================================
    #                             DELETE DATASETS
    # ========================================================================

    def test_dataset_manager__delete(self):
        """Test that the manager correctly deletes a dataset."""
        raise NotImplementedError

    def test_dataset_manager__delete_missing(self):
        """Test that the manager errors when there is no dataset to delete."""
        raise NotImplementedError
