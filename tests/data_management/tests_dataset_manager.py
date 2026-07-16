"""Tests for the DatasetManager."""

import pickle
import unittest
from pathlib import Path
from typing import List, Set, Tuple

import pyfakefs.fake_filesystem_unittest
from eliot.testing import capture_logging

from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.dataset_management.dataset_info import DatasetInfo
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from ml_on_apx.dataset_management.tree import TreeNode
from ml_on_apx.labelling import Label, Labels
from ml_on_apx.modes import Mode


class _MockDataset(Dataset):
    def __init__(self, components: Set[Tuple[Path, int]]) -> None:
        self._components = components

    def __eq__(self, other: object) -> bool:
        if type(other) is not _MockDataset:
            return False
        return self._components == other._components

    @classmethod
    def create(cls, components: Set[Tuple[Path, int]]) -> "Dataset":
        return _MockDataset(components)


class TestDatasetManager(
    unittest.TestCase, pyfakefs.fake_filesystem_unittest.TestCaseMixin
):
    """Tests for the DatasetManager."""

    def setUp(self) -> None:
        """Set up each test case."""
        self.setUpPyfakefs()
        self.data_path = Path("data")
        self.mode_path = self.data_path / Mode.Testing.value
        self.root_file_path = self.mode_path / DatasetManager.ROOT_DIR_NAME
        self.sets_path = self.mode_path / DatasetManager.SETS_DIR_NAME
        self.set_info_path = self.data_path / (
            Mode.Testing.value
            + DatasetManager.SET_INFO_NAME
            + DatasetManager.PICKLE_SUFFIX
        )

        self.manager = DatasetManager(self.data_path, Mode.Testing, _MockDataset)

        self.sources: List[Path] = [
            Path("source_1.root"),
            Path("source_2.root"),
            Path("source_3.root"),
            Path("group_a/source_a_1.root"),
            Path("group_a/source_a_2.root"),
            Path("group_b/source_b_1.root"),
            Path("group_b/source_b_2.root"),
        ]

        self.labels = Labels([Label("a"), Label("b"), Label("c")])

        self.labeling = [
            (self.sources[0], Label("a")),
            (self.sources[1], Label("b")),
            (self.sources[2], Label("c")),
            (self.sources[3], Label("a")),
            (self.sources[4], Label("a")),
            (self.sources[5], Label("b")),
            (self.sources[6], Label("b")),
        ]

        self.dataset_infos = {
            "full": DatasetInfo(self.labels, self.labeling),
            "individual": DatasetInfo(self.labels, self.labeling[0:3]),
            "group_a": DatasetInfo(self.labels, self.labeling[3:5]),
            "group_b": DatasetInfo(self.labels, self.labeling[5:7]),
            "groups": DatasetInfo(self.labels, self.labeling[3:7]),
        }

        self.mode_path.mkdir(parents=True)

    @capture_logging
    def set_up_sources(self) -> None:
        """Set up the sources in a valid way."""
        self.root_file_path.mkdir(parents=True)
        (self.root_file_path / "group_a").mkdir()
        (self.root_file_path / "group_b").mkdir()
        for source in self.sources:
            (self.root_file_path / source).touch()

    @capture_logging
    def set_up_dataset_info_pickle(self) -> None:
        """Set up the sources in a valid way."""
        with open(self.set_info_path, mode="wb") as target:
            pickle.dump(self.dataset_infos, target)

    @capture_logging
    def set_up_dataset_pickles(self) -> None:
        """Set up valid dataset pickles."""
        self.sets_path.mkdir(parents=True)
        for name, set_info in self.dataset_infos.items():
            path = self.manager._get_dataset_path(name)
            with open(path, mode="wb") as file:
                pickle.dump(_MockDataset(set_info.numbered_sources), file)

    # ========================================================================
    #                                 HELPERS
    # ========================================================================

    @capture_logging
    def test_dataset_manager__datasets_in_dir(self) -> None:
        """Test that the helper method produces the proper dataset list."""
        self.set_up_dataset_info_pickle()
        self.set_up_dataset_pickles()
        for i in range(5):
            (self.sets_path / f"{i}.filetype")
        expected = list(self.dataset_infos.keys())
        result = self.manager._datasets_in_dir(self.sets_path)
        self.assertEqual(expected, result)

    @capture_logging
    def test_dataset_manager__recompile_dataset(self) -> None:
        """Test that the helper method properly remakes and repickles the dataset."""
        self.set_up_dataset_pickles()
        path = self.manager._get_dataset_path("test")
        info = DatasetInfo(self.labels, self.labeling[0:5])
        self.manager._recompile_dataset(path, info)
        expected = _MockDataset(info.numbered_sources)
        try:
            with open(path, mode="rb") as file:
                result = pickle.load(file)
        except FileNotFoundError:
            self.fail("File not actually created.")
        self.assertEqual(expected, result)

    # ========================================================================
    #                                 ENTERING
    # ========================================================================

    @capture_logging
    def test_dataset_manager__enter_missing_root_dir(self) -> None:
        """Test that the manager creates a directory for ROOT files."""
        self.sets_path.mkdir()
        self.set_up_dataset_info_pickle()
        with self.manager as _:
            self.assertTrue(
                self.root_file_path.exists() and self.root_file_path.is_dir()
            )

    @capture_logging
    def test_dataset_manager__enter_missing_sets_dir(self) -> None:
        """Test that the manager creates a directory for datasets files."""
        self.root_file_path.mkdir()
        self.set_up_dataset_info_pickle()
        with self.manager as _:
            self.assertTrue(self.sets_path.exists() and self.sets_path.is_dir())

    @capture_logging
    def test_dataset_manager__enter_file_at_root_dir(self) -> None:
        """Test that the manager errors if the ROOT file directory path is a file."""
        self.set_up_dataset_info_pickle()
        self.root_file_path.touch()
        with self.assertRaises(NotADirectoryError):
            with self.manager as _:
                pass

    @capture_logging
    def test_dataset_manager__enter_file_at_sets_dir(self) -> None:
        """Test that the manager errors if the dataset file directory path is a file."""
        self.set_up_dataset_info_pickle()
        self.sets_path.touch()
        with self.assertRaises(NotADirectoryError):
            with self.manager as _:
                pass

    @capture_logging
    def test_dataset_manager__enters_and_exits_fine(self) -> None:
        """Test that the manager successfully operates in a normal scenario."""
        with self.manager as _:
            pass

    # ========================================================================
    #                                 SOURCES
    # ========================================================================

    @capture_logging
    def test_dataset_manager__get_sources(self) -> None:
        """Test that the manager returns the correct tree of available sources."""
        self.set_up_sources()
        expected = TreeNode(Mode.Testing.value)
        expected.add_child(TreeNode("source_1"))
        expected.add_child(TreeNode("source_2"))
        expected.add_child(TreeNode("source_3"))
        group_a = TreeNode("group_a")
        group_a.add_child(TreeNode("source_a_1"))
        group_a.add_child(TreeNode("source_a_2"))
        group_b = TreeNode("group_b")
        group_b.add_child(TreeNode("source_b_1"))
        group_b.add_child(TreeNode("source_b_2"))
        expected.add_child(group_a)
        expected.add_child(group_b)

        with self.manager as manager:
            result = manager.sources
        self.assertEqual(expected, result)

    # ========================================================================
    #                              CREATE DATASET
    # ========================================================================

    @capture_logging
    def test_dataset_manager__create(self) -> None:
        """Test that the manager can create a new dataset."""
        set_name = "new"
        self.set_up_dataset_pickles()
        self.assertFalse(self.manager._get_dataset_path(set_name).exists())
        with self.manager as manager:
            manager.create_dataset(set_name, self.dataset_infos["full"])
        self.assertTrue(self.manager._get_dataset_path(set_name).exists())

    @capture_logging
    def test_dataset_manager__create_existing(self) -> None:
        """Test that the manager errors when a dataset with that name already exists."""
        self.set_up_dataset_pickles()
        self.set_up_dataset_info_pickle()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.create_dataset("full", self.dataset_infos["full"])

    # ========================================================================
    #                              READ DATASETS
    # ========================================================================

    @capture_logging
    def test_dataset_manager__names(self) -> None:
        """Test that the manager delivers the correct list of available datasets."""
        self.set_up_dataset_pickles()
        self.set_up_dataset_info_pickle()
        expected = set(self.dataset_infos.keys())
        with self.manager as manager:
            result = manager.dataset_names
        self.assertEqual(expected, result)

    @capture_logging
    def test_dataset_manager__get_dataset_info(self) -> None:
        """Test that the manager delivers the correct dataset_info object."""
        set_name = "group_a"
        self.set_up_dataset_info_pickle()
        self.set_up_dataset_pickles()
        expected = self.dataset_infos[set_name]
        with self.manager as manager:
            result = manager.get_dataset_info(set_name)
        self.assertEqual(expected, result)

    @capture_logging
    def test_dataset_manager__get_dataset_info_missing(self) -> None:
        """Test that the manager errors when there is no dataset_info object."""
        self.set_up_dataset_info_pickle()
        self.set_up_dataset_pickles()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.get_dataset_info("absolutely positively does not exist")

    @capture_logging
    def test_dataset_manager__get_dataset(self) -> None:
        """Test that the manager delivers the correct dataset object."""
        set_name = "individual"
        self.set_up_dataset_info_pickle()
        self.set_up_dataset_pickles()
        with open(self.manager._get_dataset_path(set_name), mode="rb") as file:
            expected: _MockDataset = pickle.load(file)
        with self.manager as manager:
            result = manager.get_dataset(set_name)
        self.assertEqual(expected, result)

    @capture_logging
    def test_dataset_manager__get_dataset_group(self) -> None:
        """Test that the manager delivers the correct dataset object."""
        set_name = "group_a"
        self.set_up_dataset_info_pickle()
        self.set_up_dataset_pickles()
        with open(self.manager._get_dataset_path(set_name), mode="rb") as file:
            expected: _MockDataset = pickle.load(file)
        with self.manager as manager:
            result = manager.get_dataset(set_name)
        self.assertEqual(expected, result)

    @capture_logging
    def test_dataset_manager__get_dataset_missing(self) -> None:
        """Test that the manager errors when there is no dataset object."""
        self.set_up_dataset_info_pickle()
        self.set_up_dataset_pickles()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.get_dataset_info("absolutely positively does not exist")

    # ========================================================================
    #                             UPDATE DATASETS
    # ========================================================================

    @capture_logging
    def test_dataset_manager__update(self) -> None:
        """Test that the manager correctly updates a dataset's information."""
        new_set_name = "group_b"
        old_set_name = "group_a"
        self.set_up_dataset_info_pickle()
        self.set_up_dataset_pickles()
        expected = self.dataset_infos[new_set_name]
        with self.manager as manager:
            manager.update_dataset(old_set_name, expected)
            result = manager.get_dataset_info(old_set_name)
        self.assertEqual(expected, result)

    @capture_logging
    def test_dataset_manager__update_missing(self) -> None:
        """Test that the manager errors when there is no dataset to update."""
        dummy_set_name = self.dataset_infos["full"]
        self.set_up_dataset_info_pickle()
        self.set_up_dataset_pickles()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.update_dataset(
                    "absolutely positively does not exist", dummy_set_name
                )

    @capture_logging
    def test_dataset_manager__rename_mv_file(self) -> None:
        """Test that the manager correctly renames a dataset file."""
        set_name_from = "full"
        set_name_to = "complete"
        self.set_up_dataset_pickles()
        self.set_up_dataset_info_pickle()

        self.assertTrue(self.manager._get_dataset_path(set_name_from).exists())
        self.assertFalse(self.manager._get_dataset_path(set_name_to).exists())
        with self.manager as manager:
            manager.rename_dataset(set_name_from, set_name_to)
        self.assertTrue(self.manager._get_dataset_path(set_name_to).exists())
        self.assertFalse(self.manager._get_dataset_path(set_name_from).exists())

    @capture_logging
    def test_dataset_manager__rename_update_key(self) -> None:
        """Test that the manager correctly changes the dataset key."""
        set_name_from = "full"
        set_name_to = "complete"
        self.set_up_dataset_pickles()
        self.set_up_dataset_info_pickle()
        with self.manager as manager:
            expected = manager.get_dataset_info(set_name_from)
            with self.assertRaises(ValueError):
                manager.get_dataset_info(set_name_to)
            manager.rename_dataset(set_name_from, set_name_to)
            result = manager.get_dataset_info(set_name_to)
            with self.assertRaises(ValueError):
                manager.get_dataset_info(set_name_from)
        self.assertEqual(expected, result)

    @capture_logging
    def test_dataset_manager__rename_missing(self) -> None:
        """Test that the manager errors when there is no dataset to rename."""
        dummy_set_name = "definitely does not exist"
        self.set_up_dataset_info_pickle()
        self.set_up_dataset_pickles()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.rename_dataset(
                    "absolutely positively does not exist", dummy_set_name
                )

    @capture_logging
    def test_dataset_manager__rename_to_existing(self) -> None:
        """Test that the manager errors when the target already exists."""
        set_name_from = "full"
        set_name_to = "group_b"
        self.set_up_dataset_pickles()
        self.set_up_dataset_info_pickle()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.rename_dataset(set_name_from, set_name_to)

    @capture_logging
    def test_dataset_manager__rename_recompile(self) -> None:
        """Test that the manager correctly moves a recompilation request."""
        set_name_from = "full"
        set_name_to = "complete"
        set_name_target = self.dataset_infos["group_a"]
        self.set_up_dataset_pickles()
        self.set_up_dataset_info_pickle()

        expected = _MockDataset(set_name_target.numbered_sources)
        with self.manager as manager:
            old = manager.get_dataset(set_name_from)
            manager.update_dataset(set_name_from, set_name_target)
            manager.rename_dataset(set_name_from, set_name_to)
        with self.manager as manager:
            new = manager.get_dataset(set_name_to)

        self.assertNotEqual(old, new)
        self.assertEqual(expected, new)

    # ========================================================================
    #                             DELETE DATASETS
    # ========================================================================

    @capture_logging
    def test_dataset_manager__delete_file(self) -> None:
        """Test that the manager correctly deletes a dataset pickle."""
        set_to_delete = "full"
        self.set_up_dataset_pickles()
        self.set_up_dataset_info_pickle()

        self.assertTrue(self.manager._get_dataset_path(set_to_delete).exists())
        with self.manager as manager:
            manager.delete_dataset(set_to_delete)
        self.assertFalse(self.manager._get_dataset_path(set_to_delete).exists())

    @capture_logging
    def test_dataset_manager__delete_info(self) -> None:
        """Test that the manager correctly deletes a dataset info entry."""
        set_to_delete = "full"
        self.set_up_dataset_pickles()
        self.set_up_dataset_info_pickle()
        with self.manager as manager:
            manager.get_dataset_info(set_to_delete)
            manager.delete_dataset(set_to_delete)
            with self.assertRaises(ValueError):
                manager.get_dataset_info(set_to_delete)

    @capture_logging
    def test_dataset_manager__delete_missing(self) -> None:
        """Test that the manager errors when there is no dataset to delete."""
        self.set_up_dataset_info_pickle()
        self.set_up_dataset_pickles()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.delete_dataset("absolutely positively does not exist")
