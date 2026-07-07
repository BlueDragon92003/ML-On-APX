"""The tree of sources the user sees, and associated data."""

import enum
from pathlib import Path

from rich.style import Style
from rich.text import Text
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from ml_on_apx.labelling import Label
from ml_on_apx.logging import log_call


class SourceTreeWidget(Tree["SourceTreeData"]):
    """The widget mounted to the DOM."""

    class MissingLabelError(Exception):
        """An exception raised if a set of labeled sources is missing a label."""

        def __init__(self, path: Path, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            """Create a new MissingLabelError.

            Args:
                path (Path): The path that is missing a label.
                args: Arguments to pass to the base Exception class.
                kwargs: Keyword arguments to pass to the base exception class.

            """
            super().__init__(*args, **kwargs)
            self.path = path

    def render_label(
        self, node: TreeNode["SourceTreeData"], base_style: Style, style: Style
    ) -> Text:
        """Render the label for a node in the source tree.

        Overrides the super method in textual.widgets.Tree.

        Args:
            node (TreeNode[SourceTreeData]): A tree node.
            base_style (Style): The base style of the widget.
            style (Style): The additional style for the label.

        Returns:
            Text: A Rich Text object containing the label.

        """
        if node.data is not None:
            text = Text(node.data.get_name())
            if node.data._is_directory:
                text = Text.assemble(text, "/")
            if (
                node.data.inclusion == node.data.InclusionType.DIRECTLY_INCLUDED
                and node.data._label is not None
            ):
                text = Text.assemble(text, f" [{node.data.get_label()}]")
            node._label = text
            style = style + node.data.get_style(self.app.get_css_variables())
        return super().render_label(node, base_style, style)

    @log_call(action_type="data:app:source_tree:get_sources")
    def get_labeled_sources(self) -> set[tuple[Path, Label]]:
        """Return a set of labeled sources from the information stored in the tree.

        Returns:
            set[tuple[Path, Label]]: A set of labled sources.

        """
        labeled_sources: set[tuple[Path, Label]] = set()
        for child in self.root.children:
            labeled_sources.update(self.get_labeled_sources_from_node(child))
        return labeled_sources

    @log_call(action_type="data:app:source_tree:lab_src_from_here")
    def get_labeled_sources_from_node(
        self, node: TreeNode["SourceTreeData"]
    ) -> set[tuple[Path, Label]]:
        """Return a set of labled sources originating from this node.

        Args:
            node (TreeNode[SourceTreeData]): The node to get labeled sources from.

        Raises:
            MissingLabelError: If the node or an included descendant do not have a
                label set.

        Returns:
            set[tuple[Path, Label]]: A set of labled sources produced from this node and
                its descendants.

        """
        assert node.data is not None
        labeled_sources: set[tuple[Path, Label]] = set()
        if node.data.inclusion == node.data.InclusionType.DIRECTLY_INCLUDED:
            if node.data.is_directory():
                for path in self.get_paths_from_node(node):
                    label = node.data.get_label()
                    if label is None:
                        raise self.MissingLabelError(
                            path, f"Label not set for source `{path}`!"
                        )
                    labeled_sources.add((path, label))
            else:
                label = node.data.get_label()
                path = node.data.get_path()
                if label is None:
                    raise self.MissingLabelError(
                        path, f"Label not set for source `{path}`!"
                    )
                labeled_sources.add((path, label))

        return labeled_sources

    @log_call(action_type="data:app:source_tree:paths_from_here")
    def get_paths_from_node(self, node: TreeNode["SourceTreeData"]) -> list[Path]:
        """Get a list of paths from an included directory node.

        Args:
            node (TreeNode[SourceTreeData]): The node to extract paths from.

        Returns:
            list[Path]: A list of sources this directory causes to be included.

        """
        assert node.data is not None
        path_list: list[Path] = []
        if node.data.is_directory():
            for child in node.children:
                path_list.extend(self.get_paths_from_node(child))
        else:
            path_list.append(node.data.get_path())
        return path_list


class SourceTreeData(object):
    """The data held by a node in the source tree."""

    class NotASourceError(Exception):
        """Raised when a node represents a directory, not a source."""

    class InclusionType(enum.Enum):
        """How this node is included.

        Can be directly, not included, or via a parent (ancestrally.)
        """

        NOT_INCLUDED = 0
        ANCENSTOR_INCLUDED = 1
        DIRECTLY_INCLUDED = 2

    def __init__(self) -> None:
        """Create a new data object for a source tree node.

        Please use the classmethods `new_leaf_data` or `new_directory_data`, not the
            `__init__` method directly.
        """
        self._is_directory: bool = False
        self._descendant_error: bool = False

        self.inclusion: self.InclusionType = self.InclusionType.NOT_INCLUDED

        self._path: Path | None = None
        self._label: Label | None = None
        self._name: str

    @classmethod
    @log_call(action_type="data:app:source_tree_data:new_leaf")
    def new_leaf_data(
        cls,
        name: str,
        path: Path,
    ) -> "SourceTreeData":
        """Create a new source tree node data object that represents a source.

        Args:
            name (str): The name of the source.
            path (Path): The path at which the source can be found.

        Returns:
            SourceTreeData: The data source tree data object.

        """
        data = cls()
        data._name = name
        data._path = path
        return data

    @classmethod
    @log_call(action_type="data:app:source_tree_data:new_dir")
    def new_directory_data(
        cls,
        name: str,
    ) -> "SourceTreeData":
        """Create a new source tree node data object that represents a directory.

        Args:
            name (str): The name of the directory.

        Returns:
            SourceTreeData: The data source tree data object.

        """
        data = cls()
        data._name = name
        data._is_directory = True
        data._descendant_error = False
        return data

    @log_call(action_type="data:app:source_tree_data:label_style")
    def get_style(self, theme: dict[str, str]) -> Style:
        """Generate the style the node with this data's label should use.

        Args:
            theme (dict[str, str]): A mapping between the current theme's CSS variables
                and the colors they represent.

        Returns:
            Style: The style the label should use.

        """
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

    @log_call(action_type="data:app:source_tree_data:get_dir")
    def is_directory(self) -> bool:
        """Return true if the node with this data is a directory, not a source."""
        return self._is_directory

    @log_call(action_type="data:app:source_tree_data:get_name")
    def get_name(self) -> str:
        """Get the name of the node with this data."""
        return self._name

    @log_call(action_type="data:app:source_tree_data:get_path")
    def get_path(self) -> Path:
        """Get the name of the node with this data.

        Raises:
            NotASourceError: If the node with this data is a directory,
                not a source.

        Returns:
            Path: The path at which the source can be found.

        """
        if self._is_directory:
            raise self.NotASourceError()
        assert self._path is not None
        return self._path

    @log_call(action_type="data:app:source_tree_data:get_label")
    def get_label(self) -> Label | None:
        """Get the machine learning label of the node with this data."""
        return self._label

    @log_call(action_type="data:app:source_tree_data:st_label")
    def set_label(self, label: Label | None) -> None:
        """Set the machine learning label of the node with this data."""
        self._label = label

    @log_call(action_type="data:app:source_tree_data:rst_error")
    def reset_descendant_error(self) -> None:
        """Unset if this node has any descendants with errors."""
        self._descendant_error = False

    @log_call(action_type="data:app:source_tree_data:set_error")
    def set_descendant_has_error(self) -> None:
        """Set if this node has any descendants with errors."""
        self._descendant_error = True

    @log_call(action_type="data:app:source_tree_data:has_error")
    def has_error(self) -> bool:
        """Return True if this node or a descendant has an error."""
        if self.inclusion == self.InclusionType.DIRECTLY_INCLUDED:
            return self._label is None
        elif self._is_directory:
            return self._descendant_error
        else:
            return False
