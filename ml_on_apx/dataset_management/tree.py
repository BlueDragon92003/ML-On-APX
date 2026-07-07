"""A tree representing a directory structure of sources."""

from pathlib import Path
from typing import Iterable, List

from ml_on_apx.dataset_management import _TREE
from ml_on_apx.logging import log_call


class TreeNode:
    """A tree representing a directory structure of sources."""

    def __init__(self, name: str) -> None:
        """Create a new nodw.

        Args:
            name (str): The name of this node.

        """
        self._name: str = name
        self._children: List["TreeNode"] = []

    @log_call(action_type="get_name" @ _TREE)
    def get_name(self) -> str:
        """Get the name for this node."""
        return self._name

    @log_call(action_type="get_children" @ _TREE)
    def get_children(self) -> List["TreeNode"]:
        """Get this node's children."""
        return self._children

    @log_call(action_type="add_child" @ _TREE)
    def add_child(self, child: "TreeNode") -> None:
        """Add a child node to this node.

        Args:
            child (TreeNode): The child node to add.

        """
        self._children.append(child)

    def __eq__(self, other: object) -> bool:
        """Compare this tree with another object.

        Args:
            other (object): Another object to compare to.

        Returns:
            bool: If the other object is a tree AND stores the same hierarchy as this
                tree.

        """
        if type(other) is not TreeNode:
            return False
        if self.get_name() != other.get_name():
            return False
        children1 = self.get_children()
        children2 = other.get_children()
        if len(children1) != len(children2):
            return False
        for child1 in children1:
            match_found = False
            for child2 in children2:
                if child1 == child2:
                    match_found = True
            if not match_found:
                return False
        return True

    @staticmethod
    @log_call(action_type="from_fs" @ _TREE)
    def from_filesystem(name: str, sources: Iterable[Path]) -> "TreeNode":
        """Produce a tree of ROOT files and parent directories.

        The name of each node is the name of the directory, or the stem of the root
        file.

        Args:
            name (str): The name the root node of this tree should have
            sources (Iterable[Path]): A list of paths this node contains that should be
                checked.

        Returns:
            TreeNode: A tree representing root files within their directory structure.

        """
        out = TreeNode(name)
        for source in sources:
            if not source.exists():
                continue
            elif source.is_dir():
                out.add_child(TreeNode.from_filesystem(source.name, source.iterdir()))
            elif source.suffix == ".root":
                out.add_child(TreeNode(source.stem))
        return out
