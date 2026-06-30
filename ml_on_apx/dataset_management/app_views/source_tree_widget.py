from textual.widgets.tree import TreeNode
from textual.widgets import Tree
import enum
from rich.style import Style
from rich.text import Text
from ml_on_apx.labelling import Label
from pathlib import Path


class SourceTreeWidget(Tree["SourceTreeData"]):
    def render_label(
        self, node: TreeNode["SourceTreeData"], base_style: Style, style: Style
    ) -> Text:
        if node.data is not None:
            text = Text(node.data._name)
            if node.data._is_directory:
                text = Text.assemble(text, "/")
            if (
                node.data.inclusion == node.data.InclusionType.DIRECTLY_INCLUDED
                and node.data._label is not None
            ):
                text = Text.assemble(text, f" [{node._label}]")
            node._label = text
            style = style + node.data.get_style(self.app.get_css_variables())
        return super().render_label(node, base_style, style)

    def get_labeled_sources(self) -> dict[Path, Label]:
        labeled_sources: dict[Path, Label] = {}
        for child in self.root.children:
            labeled_sources.update(self.get_labeled_sources_from_node(child))
        return labeled_sources

    def get_labeled_sources_from_node(
        self, node: TreeNode["SourceTreeData"]
    ) -> set[tuple[Path, Label]]:
        assert node.data is not None
        labeled_sources: set[tuple[Path, Label]] = set()
        if node.data.inclusion == node.data.InclusionType.DIRECTLY_INCLUDED:
            if node.data.is_directory():
                for path in self.get_paths_from_node(node):
                    label = node.data.get_label()
                    if label is None:
                        raise ValueError(f"Label not set for source `{path}`!")
                    labeled_sources.add((path, label))
            else:
                label = node.data.get_label()
                if label is None:
                    raise ValueError(
                        f"Label not set for source `{node.data.get_path()}`!"
                    )
                labeled_sources.add((node.data.get_path(), label))

        return labeled_sources

    def get_paths_from_node(self, node: TreeNode["SourceTreeData"]) -> list[Path]:
        assert node.data is not None
        path_list: list[Path] = []
        if node.data.is_directory():
            for child in node.children:
                path_list.extend(self.get_paths_from_node(child))
        else:
            path_list.append(node.data.get_path())
        return path_list


class SourceTreeData(object):
    class NotASourceException(Exception):
        pass

    class NotADirectoryException(Exception):
        pass

    class InclusionType(enum.Enum):
        NOT_INCLUDED = 0
        ANCENSTOR_INCLUDED = 1
        DIRECTLY_INCLUDED = 2

    def __init__(self):
        self._is_directory: bool = False
        self._descendant_error: bool = False

        self.inclusion: self.InclusionType = self.InclusionType.NOT_INCLUDED

        self._path: Path | None = None
        self._label: Label | None = None
        self._name: str

    @classmethod
    def new_leaf_data(
        cls,
        name: str,
        path: Path,
    ) -> "SourceTreeData":
        data = cls()
        data._name = name
        data._path = path
        return data

    @classmethod
    def new_directory_data(
        cls,
        name: str,
    ) -> "SourceTreeData":
        data = cls()
        data._name = name
        data._is_directory = True
        data._descendant_error = False
        return data

    def get_style(self, theme: dict[str, str]) -> Style:
        if self.inclusion == self.InclusionType.ANCENSTOR_INCLUDED:
            # included parents
            style = Style(color=theme["foreground-darken-3"])
        elif self.inclusion == self.InclusionType.DIRECTLY_INCLUDED:
            if self._label is not None:
                # included, no error
                style = Style(color=theme["text-success"], bold=True)
            else:
                # included, error
                style = Style(color=theme["text-error"], bold=True)
        elif self._is_directory:
            if self._descendant_error:
                # No included children
                style = Style(color=theme["text-warning"])
            else:
                # errors in included children
                style = Style(color=theme["foreground"])
        else:
            # Not included
            style = Style(color=theme["foreground"])
        return style

    def is_directory(self) -> bool:
        return self._is_directory

    def get_path(self) -> Path:
        if self._is_directory:
            raise self.NotASourceException()
        assert self._path is not None
        return self._path

    def get_label(self) -> Label | None:
        return self._label

    def set_label(self, label: Label | None):
        self._label = label

    def reset_descendant_error(self):
        self._descendant_error = False

    def set_descendant_has_error(self):
        self._descendant_error = True

    def has_error(self) -> bool:
        if self.inclusion == self.InclusionType.DIRECTLY_INCLUDED:
            return self._label is None
        elif self._is_directory:
            return self._descendant_error
        else:
            return False


"""

# Leaf node: ROOT Dataset
States:
- Not included              ($foreground)
- Ancestor included         ($foreground.muted)
- Included with label       ($text-success, bold)
- Included, label not set   ($text-error, bold)
Data:
- Path to source
- Included/not included
- Label, if included
Messages:
- Selected
- LabelSet

# Node: Directory
States:
- No decendant included                         ($foreground)
- Ancestor included                             ($foreground.muted)
- (Some) decendants included without error      ($foreground, bold)
- (Some) decendants included with error         ($text-warning, bold)
- Directly included with label                  ($text-success, bold)
- Directly included, label not set              ($text-error, bold)
Data:
- Path to directory
- Directly included/not directly included
- Label, if directly included
Messages:
- Selected
- LabelSet
Message Handlers:
- leaf.Selected
- leaf.


"""
