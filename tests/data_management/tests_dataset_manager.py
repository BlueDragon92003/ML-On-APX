import pyfakefs.fake_filesystem_unittest
import unittest


class TestDatasetManager(
    unittest.TestCase, pyfakefs.fake_filesystem_unittest.TestCaseMixin
):
    raise NotImplementedError

    def setUp():
        raise NotImplementedError

    def test_dataset_manager__instantiation(self):
        raise NotImplementedError

    def test_dataset_manager__enters_and_exits_fine(self):
        raise NotImplementedError

    def test_dataset_manager__create_works(self):
        raise NotImplementedError

    def test_dataset_manager__create_value_error(self):
        raise NotImplementedError

    def test_dataset_manager__names(self):
        raise NotImplementedError

    def test_dataset_manager__get_item(self):
        raise NotImplementedError

    def test_dataset_manager__get_missing(self):
        raise NotImplementedError

    def test_dataset_manager__update(self):
        raise NotImplementedError

    def test_dataset_manager__update_missing(self):
        raise NotImplementedError

    def test_dataset_manager__rename(self):
        raise NotImplementedError

    def test_dataset_manager__rename_missing(self):
        raise NotImplementedError

    def test_dataset_manager__delete(self):
        raise NotImplementedError

    def test_dataset_manager__delete_missing(self):
        raise NotImplementedError

    def test_dataset_manager__get_dataset_path(self):
        raise NotImplementedError

    def test_dataset_manager__recompile_dataset(self):
        raise NotImplementedError

    def test_dataset_manager__get_sources(self):
        raise NotImplementedError

    def test_dataset_manager__recompiles_on_exit(self):
        raise NotImplementedError

    def test_dataset_manager__enter_missing_root_dir(self):
        raise NotImplementedError

    def test_dataset_manager__enter_missing_sets_dir(self):
        raise NotImplementedError

    def test_dataset_manager__enter_file_at_root_dir(self):
        raise NotImplementedError

    def test_dataset_manager__enter_file_at_sets_dir(self):
        raise NotImplementedError
