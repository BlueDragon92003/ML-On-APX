import unittest
from pathlib import Path
import pyfakefs
import pyfakefs.fake_filesystem_unittest
from ml_on_apx.dataset_management.tree import TreeNode


class TestTreeFromFS(
    unittest.TestCase, pyfakefs.fake_filesystem_unittest.TestCaseMixin
):
    def setUp(self):
        self.setUpClassPyfakefs()

    def test_tree_from_fs(self):
        NAME = "root"
        DEPTH = 3
        PATH = Path(NAME)
        self.create_fake_files(PATH, depth=DEPTH)
        expected = self.create_expected_tree(NAME, depth=DEPTH)
        result = TreeNode.from_filesystem(NAME, PATH.iterdir())
        self.assertTrue(
            self.compare_trees(expected, result),
            msg="Trees not equivalent.\n\n"
            + self.print_tree(expected)
            + "\n"
            + self.print_tree(result),
        )

    def print_tree(self, tree: TreeNode, depth=0) -> str:
        out = ("\t" * depth) + tree.get_name() + "/\n"
        for child in tree.get_children():
            out += self.print_tree(child, depth=depth + 1)
        return out

    def compare_trees(self, tree1: TreeNode, tree2: TreeNode) -> bool:
        if tree1.get_name() != tree2.get_name():
            return False
        children1 = tree1.get_children()
        children2 = tree2.get_children()
        if len(children1) != len(children2):
            return False
        for child1 in children1:
            match_found = False
            for child2 in children2:
                if self.compare_trees(child1, child2):
                    match_found = True
            if not match_found:
                return False
        return True

    def create_expected_tree(self, name: str, depth: int) -> TreeNode:
        out = TreeNode(name)
        for i in range(depth):
            out.add_child(self.create_expected_tree(f"f{i}.root", depth=i))
        return out

    def create_fake_files(self, path: Path, depth: int):
        if depth == 0:
            path.touch()
            return
        path.mkdir()
        for i in range(depth):
            self.create_fake_files(path / f"f{i}.root", depth=i)
