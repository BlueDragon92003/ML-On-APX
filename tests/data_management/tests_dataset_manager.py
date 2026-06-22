from ml_on_apx.dataset_management.tree import TreeNode
from ml_on_apx.labelling import Labels, Label
from ml_on_apx.dataset_management.dataset_info import DatasetInfo
import pickle
from ml_on_apx.modes import Mode
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from typing import Tuple, Set, List
from pathlib import Path
from ml_on_apx.dataset_management.dataset import Dataset
import pyfakefs.fake_filesystem_unittest
import unittest


class _MockDataset(Dataset):
    def __init__(self, components: Set[Tuple[Path, int]]):
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
    def setUp(self):
        self.setUpPyfakefs()
        self.data_path = Path("data")
        self.mode_path = self.data_path / Mode.Testing.value
        self.root_file_path = self.mode_path / DatasetManager.ROOT_DIR_NAME
        self.sets_path = self.mode_path / DatasetManager.SETS_DIR_NAME
        self.set_info_path = (
            self.data_path
            / f"{Mode.Testing.value}{DatasetManager.SET_INFO_NAME}{DatasetManager.PICKLE_SUFFIX}"
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

    def setUpSources(self):
        self.root_file_path.mkdir(parents=True)
        (self.root_file_path / "group_a").mkdir()
        (self.root_file_path / "group_b").mkdir()
        for source in self.sources:
            (self.root_file_path / source).touch()

    def setUpDatasetInfoPickle(self):
        with open(self.set_info_path, mode="wb") as target:
            pickle.dump(self.dataset_infos, target)

    def setUpDatasetPickles(self):
        self.sets_path.mkdir(parents=True)
        for name, set_info in self.dataset_infos.items():
            path = self.manager._get_dataset_path(name)
            with open(path, mode="wb") as file:
                pickle.dump(_MockDataset(set_info.get_numbered_sources()), file)

    # ========================================================================
    #                                 HELPERS
    # ========================================================================

    def test_dataset_manager__datasets_in_dir(self):
        """Test that the helper method produces the proper dataset list."""
        self.setUpDatasetInfoPickle()
        self.setUpDatasetPickles()
        for i in range(5):
            (self.sets_path / f"{i}.filetype")
        expected = list(self.dataset_infos.keys())
        result = self.manager._datasets_in_dir(self.sets_path)
        self.assertEqual(expected, result)

    def test_dataset_manager__recompile_dataset(self):
        """Test that the helper method properly remakes and repickles the dataset."""
        self.setUpDatasetPickles()
        path = self.manager._get_dataset_path("test")
        info = DatasetInfo(self.labels, self.labeling[0:5])
        self.manager._recompile_dataset(path, info)
        expected = _MockDataset(info.get_numbered_sources())
        try:
            with open(path, mode="rb") as file:
                result = pickle.load(file)
        except FileNotFoundError:
            self.fail("File not actually created.")
        self.assertEqual(expected, result)

    # ========================================================================
    #                                 ENTERING
    # ========================================================================

    def test_dataset_manager__enter_missing_root_dir(self):
        """Test that the manager creates a directory for ROOT files."""
        self.sets_path.mkdir()
        self.setUpDatasetInfoPickle()
        with self.manager as _:
            self.assertTrue(
                self.root_file_path.exists() and self.root_file_path.is_dir()
            )

    def test_dataset_manager__enter_missing_sets_dir(self):
        """Test that the manager creates a directory for datasets files."""
        self.root_file_path.mkdir()
        self.setUpDatasetInfoPickle()
        with self.manager as _:
            self.assertTrue(self.sets_path.exists() and self.sets_path.is_dir())

    def test_dataset_manager__enter_file_at_root_dir(self):
        """Test that the manager errors if the ROOT file directory path is a file."""
        self.setUpDatasetInfoPickle()
        self.root_file_path.touch()
        with self.assertRaises(NotADirectoryError):
            with self.manager as _:
                pass

    def test_dataset_manager__enter_file_at_sets_dir(self):
        """Test that the manager errors if the dataset file directory path is a file."""
        self.setUpDatasetInfoPickle()
        self.sets_path.touch()
        with self.assertRaises(NotADirectoryError):
            with self.manager as _:
                pass

    def test_dataset_manager__enters_and_exits_fine(self):
        """Test that the manager successfully operates in a normal scenario."""
        with self.manager as _:
            pass

    # ========================================================================
    #                                 SOURCES
    # ========================================================================

    def test_dataset_manager__get_sources(self):
        """Test that the manager returns the correct tree of available sources."""
        self.setUpSources()
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
            result = manager.get_sources()
        self.assertEqual(expected, result)

    # ========================================================================
    #                              CREATE DATASET
    # ========================================================================

    def test_dataset_manager__create(self):
        """Test that the manager can create a new dataset."""
        NAME = "new"
        self.setUpDatasetPickles()
        self.assertFalse(self.manager._get_dataset_path(NAME).exists())
        with self.manager as manager:
            manager.create_dataset(NAME, self.dataset_infos["full"])
        self.assertTrue(self.manager._get_dataset_path(NAME).exists())

    def test_dataset_manager__create_existing(self):
        """Test that the manager errors when a dataset with that name already exists."""
        self.setUpDatasetPickles()
        self.setUpDatasetInfoPickle()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.create_dataset("full", self.dataset_infos["full"])

    # ========================================================================
    #                              READ DATASETS
    # ========================================================================

    def test_dataset_manager__names(self):
        """Test that the manager delivers the correct list of available datasets."""
        self.setUpDatasetPickles()
        self.setUpDatasetInfoPickle()
        expected = set(self.dataset_infos.keys())
        with self.manager as manager:
            result = manager.get_dataset_names()
        self.assertEqual(expected, result)

    def test_dataset_manager__get_dataset_info(self):
        """Test that the manager delivers the correct dataset_info object."""
        SET = "group_a"
        self.setUpDatasetInfoPickle()
        self.setUpDatasetPickles()
        expected = self.dataset_infos[SET]
        with self.manager as manager:
            result = manager.get_dataset_info(SET)
        self.assertEqual(expected, result)

    def test_dataset_manager__get_dataset_info_missing(self):
        """Test that the manager errors when there is no dataset_info object."""
        self.setUpDatasetInfoPickle()
        self.setUpDatasetPickles()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.get_dataset_info("absolutely positively does not exist")

    def test_dataset_manager__get_dataset(self):
        """Test that the manager delivers the correct dataset object."""
        SET = "individual"
        self.setUpDatasetInfoPickle()
        self.setUpDatasetPickles()
        with open(self.manager._get_dataset_path(SET), mode="rb") as file:
            expected: _MockDataset = pickle.load(file)
        with self.manager as manager:
            result = manager.get_dataset(SET)
        self.assertEqual(expected, result)

    def test_dataset_manager__get_dataset_group(self):
        """Test that the manager delivers the correct dataset object."""
        SET = "group_a"
        self.setUpDatasetInfoPickle()
        self.setUpDatasetPickles()
        with open(self.manager._get_dataset_path(SET), mode="rb") as file:
            expected: _MockDataset = pickle.load(file)
        with self.manager as manager:
            result = manager.get_dataset(SET)
        self.assertEqual(expected, result)

    def test_dataset_manager__get_dataset_missing(self):
        """Test that the manager errors when there is no dataset object."""
        self.setUpDatasetInfoPickle()
        self.setUpDatasetPickles()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.get_dataset_info("absolutely positively does not exist")

    # ========================================================================
    #                             UPDATE DATASETS
    # ========================================================================

    def test_dataset_manager__update(self):
        """Test that the manager correctly updates a dataset's information."""
        NEW = "group_b"
        OLD = "group_a"
        self.setUpDatasetInfoPickle()
        self.setUpDatasetPickles()
        expected = self.dataset_infos[NEW]
        with self.manager as manager:
            manager.update_dataset(OLD, expected)
            result = manager.get_dataset_info(OLD)
        self.assertEqual(expected, result)

    def test_dataset_manager__update_missing(self):
        """Test that the manager errors when there is no dataset to update."""
        DUMMY = self.dataset_infos["full"]
        self.setUpDatasetInfoPickle()
        self.setUpDatasetPickles()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.update_dataset("absolutely positively does not exist", DUMMY)

    def test_dataset_manager__rename_mv_file(self):
        """Test that the manager correctly renames a dataset file."""
        FROM = "full"
        TO = "complete"
        self.setUpDatasetPickles()
        self.setUpDatasetInfoPickle()

        self.assertTrue(self.manager._get_dataset_path(FROM).exists())
        self.assertFalse(self.manager._get_dataset_path(TO).exists())
        with self.manager as manager:
            manager.rename_dataset(FROM, TO)
        self.assertTrue(self.manager._get_dataset_path(TO).exists())
        self.assertFalse(self.manager._get_dataset_path(FROM).exists())

    def test_dataset_manager__rename_update_key(self):
        """Test that the manager correctly changes the dataset key."""
        FROM = "full"
        TO = "complete"
        self.setUpDatasetPickles()
        self.setUpDatasetInfoPickle()
        with self.manager as manager:
            expected = manager.get_dataset_info(FROM)
            with self.assertRaises(ValueError):
                manager.get_dataset_info(TO)
            manager.rename_dataset(FROM, TO)
            result = manager.get_dataset_info(TO)
            with self.assertRaises(ValueError):
                manager.get_dataset_info(FROM)
        self.assertEqual(expected, result)

    def test_dataset_manager__rename_missing(self):
        """Test that the manager errors when there is no dataset to rename."""
        DUMMY = "definitely does not exist"
        self.setUpDatasetInfoPickle()
        self.setUpDatasetPickles()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.rename_dataset("absolutely positively does not exist", DUMMY)

    def test_dataset_manager__rename_to_existing(self):
        """Test that the manager errors when the target already exists."""
        FROM = "full"
        TO = "group_b"
        self.setUpDatasetPickles()
        self.setUpDatasetInfoPickle()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.rename_dataset(FROM, TO)

    def test_dataset_manager__rename_recompile(self):
        """Test that the manager correctly moves a recompilation request."""
        FROM = "full"
        TO = "complete"
        TARGET = self.dataset_infos["group_a"]
        self.setUpDatasetPickles()
        self.setUpDatasetInfoPickle()

        expected = _MockDataset(TARGET.get_numbered_sources())
        with self.manager as manager:
            old = manager.get_dataset(FROM)
            manager.update_dataset(FROM, TARGET)
            manager.rename_dataset(FROM, TO)
        with self.manager as manager:
            new = manager.get_dataset(TO)

        self.assertNotEqual(old, new)
        self.assertEqual(expected, new)

    # ========================================================================
    #                             DELETE DATASETS
    # ========================================================================

    def test_dataset_manager__delete_file(self):
        """Test that the manager correctly deletes a dataset pickle."""
        TO_DELETE = "full"
        self.setUpDatasetPickles()
        self.setUpDatasetInfoPickle()

        self.assertTrue(self.manager._get_dataset_path(TO_DELETE).exists())
        with self.manager as manager:
            manager.delete_dataset(TO_DELETE)
        self.assertFalse(self.manager._get_dataset_path(TO_DELETE).exists())

    def test_dataset_manager__delete_info(self):
        """Test that the manager correctly deletes a dataset info entry."""
        TO_DELETE = "full"
        self.setUpDatasetPickles()
        self.setUpDatasetInfoPickle()
        with self.manager as manager:
            manager.get_dataset_info(TO_DELETE)
            manager.delete_dataset(TO_DELETE)
            with self.assertRaises(ValueError):
                manager.get_dataset_info(TO_DELETE)

    def test_dataset_manager__delete_missing(self):
        """Test that the manager errors when there is no dataset to delete."""
        self.setUpDatasetInfoPickle()
        self.setUpDatasetPickles()
        with self.assertRaises(ValueError):
            with self.manager as manager:
                manager.delete_dataset("absolutely positively does not exist")
