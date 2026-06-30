from ml_on_apx.dataset_management.app_views.source_tree_widget import (
    SourceTreeData,
    SourceTreeWidget,
)
from textual import on
from textual.widgets.tree import TreeNode
from textual.binding import Binding
from textual.containers import VerticalScroll, HorizontalGroup
from textual.widgets import (
    TabbedContent,
    Label,
    TabPane,
    Input,
    Button,
    ListView,
    ListItem,
    Tree,
    Footer,
    Header,
)
from textual.app import ComposeResult
from pathlib import Path
from ml_on_apx.labelling import Label as SourceLabel
from textual.reactive import reactive
from ml_on_apx.dataset_management.dataset_info import DatasetInfo
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from ml_on_apx.dataset_management.tree import TreeNode as SourceTreeNode
from textual.screen import Screen


class NewEditView(Screen[None]):
    BINDINGS_ = [
        ("I", "to_basic_info", "Basic info"),
        ("L", "to_labels", "Labels"),
        ("S", "to_sources", "Sources"),
        ("a", "new_label", "Add label"),
        ("d", "delete_label", "Delete label"),
        Binding("ctrl+D", "force_delete_label", show=False),  # without asking
        ("l", "set_source_label", "Set source label"),
    ]

    dataset_name: reactive[str] = reactive("")
    labels: reactive[list[SourceLabel]] = reactive([])

    # TODO consider popup on selection to get initial label?
    #   add label key would just be a "formality"
    #   then have a toggle label key?

    def __init__(
        self,
        manager: DatasetManager,
        template: DatasetInfo | None = None,
        template_name: str | None = None,
    ):
        super().__init__()
        self._manager = manager
        self.dataset_name = template_name if template_name else ""
        if template is not None:
            for label in template.get_labels():
                self.labels.append(label)
            for source, label in template.get_labeled_sources():
                # TODO
                pass

    def compose(self) -> ComposeResult:
        yield Header(
            name="New Dataset" if self.dataset_name else f"Edit {self.dataset_name}"
        )
        yield Footer()
        with TabbedContent(id="new-edit-tabs"):
            with TabPane("Basic Info", id="general-info-tab"):
                with VerticalScroll():
                    with HorizontalGroup():
                        yield Label("Name: ")
                        yield Input(
                            value=self.dataset_name,
                            placeholder="Dataset name...",
                            id="name-input",
                        )
                    with HorizontalGroup(id="new-edit-control-buttons"):
                        yield Button("Cancel", variant="default")
                        yield Button("Create", variant="success")
            with TabPane("Labels", id="labels-tab"):
                yield ListView(id="labels-list")
            with TabPane("Sources", id="sources-tab"):
                self._tree = SourceTreeWidget(
                    "/".join(self._manager.get_root_dir_path().parts) + "/",
                    id="source-tree",
                )
                self._tree.auto_expand = False
                yield self._tree

    def on_mount(self):
        self.append_nodes(
            self._tree.root,
            self._manager.get_sources(),
            Path(self._manager.ROOT_DIR_NAME),
        )
        self.remake_label_list(self.labels)

    @on(Tree.NodeSelected)
    def handle_source_selection(self, message: Tree.NodeSelected[SourceTreeData]):
        if message.node.data is None:
            message.stop()
            return
        data = message.node.data
        match data.inclusion:
            case _inclusion if _inclusion == data.InclusionType.DIRECTLY_INCLUDED:
                self.disinclude(message.node)
            case _inclusion if _inclusion == data.InclusionType.ANCENSTOR_INCLUDED:
                assert message.node.parent is not None
                self.release_children(message.node.parent)
                data.inclusion = data.InclusionType.NOT_INCLUDED
                for child in message.node.children:
                    self.disinclude(child)
            case _inclusion if _inclusion == data.InclusionType.NOT_INCLUDED:
                data.inclusion = data.InclusionType.DIRECTLY_INCLUDED
                for child in message.node.children:
                    self.include(child)
        self.update_source_tree_selections(message.node.tree.root)
        message.stop()

    def watch_labels(
        self, new_labels: list[SourceLabel], old_labels: list[SourceLabel]
    ):
        self.remake_label_list(new_labels)
        tree_node: TreeNode[SourceTreeData] = self._tree.root
        for label in old_labels:
            if label not in new_labels:
                self.remove_label_from_tree(tree_node, label)
        self.update_source_tree_selections(tree_node)

    def watch_sources(self, new_sources: dict[Path, SourceLabel | None]):
        tree = self._tree
        tree_node: TreeNode[SourceTreeData] = tree.root
        print(f"Updating sources to {new_sources.items()}")
        self.update_source_tree_selections(tree_node)

    def remake_label_list(self, labels: list[SourceLabel]):
        labels_list = self.get_widget_by_id("labels-list")
        assert type(labels_list) is ListView
        for label in labels:
            labels_list.append(ListItem(Label(f"{label}"), name=f"{label}"))

    def update_source_tree_selections(self, tree_node: TreeNode[SourceTreeData]):
        root = tree_node.tree.root
        if tree_node == root:
            for child in tree_node.children:
                self.update_source_tree_selections(child)
            return
        assert tree_node.data is not None
        tree_node.data.reset_descendant_error()
        for child in tree_node.children:
            self.update_source_tree_selections(child)
            assert child.data is not None
            if child.data.has_error():
                tree_node.data.set_descendant_has_error()
        tree_node.refresh()

    def remove_label_from_tree(
        self, tree_node: TreeNode[SourceTreeData], label_to_remove: SourceLabel
    ):
        if tree_node is not tree_node.tree.root:
            assert tree_node.data is not None
            if tree_node.data.get_label() is label_to_remove:
                tree_node.data.set_label(None)
        for child in tree_node.children:
            self.remove_label_from_tree(child, label_to_remove)

    def append_nodes(
        self,
        dest_tree_node: TreeNode[SourceTreeData],
        source_tree: SourceTreeNode,
        path_so_far: Path,
    ):
        for source_node in source_tree.get_children():
            this_path = path_so_far / source_node.get_name()
            if len(source_node.get_children()) == 0:
                new_tree_node = dest_tree_node.add_leaf(
                    f"{source_node.get_name()}",
                    data=SourceTreeData.new_leaf_data(
                        f"{source_node.get_name()}", this_path
                    ),
                )
            else:
                new_tree_node = dest_tree_node.add(
                    f"{source_node.get_name()}",
                    data=SourceTreeData.new_directory_data(f"{source_node.get_name()}"),
                )
            self.append_nodes(new_tree_node, source_node, this_path)
        dest_tree_node.expand()

    def release_children(self, node: TreeNode[SourceTreeData]):
        """Inform root ancestor to include its children, not itself directly"""
        assert node.data is not None
        if node.data.inclusion == node.data.InclusionType.ANCENSTOR_INCLUDED:
            assert node.parent is not None
            self.release_children(node.parent)
        elif node.data.inclusion == node.data.InclusionType.DIRECTLY_INCLUDED:
            node.data.inclusion = node.data.InclusionType.NOT_INCLUDED
        for child in node.children:
            assert child.data is not None
            child.data.inclusion = child.data.InclusionType.DIRECTLY_INCLUDED
            child.data.set_label(node.data.get_label())
        node.data.set_label(None)

    def include(self, node: TreeNode[SourceTreeData]):
        """Include this node ancestraly"""
        assert node.data is not None
        node.data.inclusion = node.data.InclusionType.ANCENSTOR_INCLUDED
        node.data.set_label(None)
        for child in node.children:
            self.include(child)

    def disinclude(self, node: TreeNode[SourceTreeData]):
        """This nodes inclusion ancestor was disincluded"""
        assert node.data is not None
        node.data.inclusion = node.data.InclusionType.NOT_INCLUDED
        node.data.set_label(None)
        for child in node.children:
            self.disinclude(child)
