from pathlib import Path
from typing import Iterable, List


class TreeNode:
    def __init__(self, name: str):
        self._name: str = name
        self._children: List["TreeNode"] = []

    def get_name(self) -> str:
        return self._name

    def get_children(self) -> List["TreeNode"]:
        return self._children

    def add_child(self, child: "TreeNode"):
        self._children.append(child)

    @staticmethod
    def from_filesystem(name: str, sources: Iterable[Path]) -> "TreeNode":
        out = TreeNode(name)
        for source in sources:
            if not source.exists():
                continue
            elif source.is_dir():
                out.add_child(TreeNode.from_filesystem(source.name, source.iterdir()))
            elif source.suffix == ".root":
                out.add_child(TreeNode(source.name))
        return out
