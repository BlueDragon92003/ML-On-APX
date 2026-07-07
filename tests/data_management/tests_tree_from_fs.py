"""Test that creation of a tree from a filesystem works as expected."""

import unittest
from pathlib import Path

import pyfakefs
import pyfakefs.fake_filesystem_unittest
from eliot.testing import capture_logging

from ml_on_apx.dataset_management.tree import TreeNode


class TestTreeFromFS(
    unittest.TestCase, pyfakefs.fake_filesystem_unittest.TestCaseMixin
):
    """Test that creation of a tree from a filesystem works as expected."""

    @capture_logging
    def setUp(self) -> None:
        """Set up the test cases."""
        self.setUpClassPyfakefs()

    @capture_logging
    def test_tree_from_fs(self) -> None:
        """Test the creation of a tree from a filesystem."""
        tree_root_name = "root"
        depth_to_test = 3
        tree_root_path = Path(tree_root_name)
        self.create_fake_files(tree_root_path, depth=depth_to_test)
        expected = self.create_expected_tree(tree_root_name, depth=depth_to_test)
        result = TreeNode.from_filesystem(tree_root_name, tree_root_path.iterdir())
        self.assertEqual(
            expected,
            result,
            msg="Trees not equivalent.\n\n"
            + self.print_tree(expected)
            + "\n"
            + self.print_tree(result),
        )

    @capture_logging
    def print_tree(self, tree: TreeNode, depth: int = 0) -> str:
        """Create a string representation of the tree."""
        out = ("\t" * depth) + tree.get_name() + "/\n"
        for child in tree.get_children():
            out += self.print_tree(child, depth=depth + 1)
        return out

    @capture_logging
    def create_expected_tree(
        self, name: str, depth: int, recursed: bool = False
    ) -> TreeNode:
        """Create the tree expected from the create_from_filesystem function."""
        out = TreeNode(name + (".root" if depth != 0 and recursed else ""))
        for i in range(depth):
            out.add_child(self.create_expected_tree(f"f{i}", depth=i, recursed=True))
        return out

    @capture_logging
    def create_fake_files(self, path: Path, depth: int) -> None:
        """Create fake files in the expected shape."""
        if depth == 0:
            path.touch()
            return
        path.mkdir()
        for i in range(depth):
            self.create_fake_files(path / f"f{i}.root", depth=i)
