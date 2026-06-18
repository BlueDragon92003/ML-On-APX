from ml_on_apx.dataset_management.tree import TreeNode
import unittest


class TestTree(unittest.TestCase):
    def test_tree__instantiation(self):
        TreeNode("test")

    def test_tree__get_name(self):
        NAME = "name"
        node = TreeNode(NAME)
        self.assertEqual(node.get_name(), NAME)

    def test_tree__get_no_children(self):
        node = TreeNode("NAME")
        self.assertEqual(node.get_children(), [])

    def test_tree__add_child(self):
        node = TreeNode("1")
        node.add_child(TreeNode("2"))

    def test_tree__get_children(self):
        node = TreeNode("root")
        for i in range(5):
            node.add_child(TreeNode(str(i)))
        children = node.get_children()
        self.assertEqual(len(children), 5)
        result = []
        for child in children:
            result.append(child.get_name())
        for i in range(5):
            self.assertIn(str(i), result)
