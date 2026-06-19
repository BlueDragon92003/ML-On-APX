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

    def __eq__(self, other: object) -> bool:
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
    def from_filesystem(name: str, sources: Iterable[Path]) -> "TreeNode":
        out = TreeNode(name)
        for source in sources:
            if not source.exists():
                continue
            elif source.is_dir():
                out.add_child(TreeNode.from_filesystem(source.stem, source.iterdir()))
            elif source.suffix == ".root":
                out.add_child(TreeNode(source.stem))
        return out
