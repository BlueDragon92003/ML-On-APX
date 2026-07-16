"""Most tests for the TreeNode class."""

import unittest

from eliot.testing import capture_logging

from ml_on_apx.dataset_management.tree import TreeNode


class TestTree(unittest.TestCase):
    """Most tests for the TreeNode class."""

    @capture_logging
    def test_tree__instantiation(self) -> None:
        """Test there are no errors on tree creation."""
        TreeNode("test")

    @capture_logging
    def test_tree__get_name(self) -> None:
        """Test the get_name function."""
        name = "name"
        node = TreeNode(name)
        self.assertEqual(node.name, name)

    @capture_logging
    def test_tree__get_no_children(self) -> None:
        """Test that the get_children works as expected without children."""
        node = TreeNode("NAME")
        self.assertEqual(node.children, [])

    @capture_logging
    def test_tree__add_child(self) -> None:
        """Test adding a child to the tree does not error."""
        node = TreeNode("1")
        node.add_child(TreeNode("2"))

    @capture_logging
    def test_tree__get_children(self) -> None:
        """Test that getting a child from the tree functions."""
        node = TreeNode("root")
        for i in range(5):
            node.add_child(TreeNode(str(i)))
        children = node.children
        self.assertEqual(len(children), 5)
        result = []
        for child in children:
            result.append(child.name)
        for i in range(5):
            self.assertIn(str(i), result)
