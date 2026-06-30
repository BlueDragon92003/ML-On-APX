from ml_on_apx.dataset_management.app_views.source_tree_data import SourceTreeData
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
    sources: reactive[dict[Path, SourceLabel | None]] = reactive({})

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
                self.sources.update({source: label})

    def compose(self) -> ComposeResult:
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
                self._tree = Tree[SourceTreeData](
                    "/".join(self._manager.get_root_dir_path().parts), id="source-tree"
                )
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
        if data.included:
            self.sources.pop(data.get_path())
            data.included = False
        else:
            self.sources.update({data.get_path(): data.get_label()})
            data.included = True
        self.update_source_tree_selections(message.node, self.app.get_css_variables())
        message.stop()
        # self.app.theme = "nord" if self.app.theme == "gruvbox" else "gruvbox"

    def watch_labels(
        self, new_labels: list[SourceLabel], old_labels: list[SourceLabel]
    ):
        self.remake_label_list(new_labels)
        tree_node: TreeNode[SourceTreeData] = self._tree.root
        for label in old_labels:
            if label not in new_labels:
                self.remove_label_from_tree(tree_node, label)
        self.update_source_tree_selections(tree_node, self.app.get_css_variables())

    def watch_sources(self, new_sources: dict[Path, SourceLabel | None]):
        tree = self._tree
        tree_node: TreeNode[SourceTreeData] = tree.root
        print(f"Updating sources to {new_sources.items()}")
        self.update_source_tree_selections(tree_node, self.app.get_css_variables())

    def remake_label_list(self, labels: list[SourceLabel]):
        labels_list = self.get_widget_by_id("labels-list")
        assert type(labels_list) is ListView
        for label in labels:
            labels_list.append(ListItem(Label(f"{label}"), name=f"{label}"))

    def update_source_tree_selections(
        self, tree_node: TreeNode[SourceTreeData], theme: dict[str, str]
    ):
        # TODO recursive for each node:
        #   Change color of text (hidden if unselected, green & bold if selected, red & bold if selected without label)
        #   Change label to "name (label)"
        root = tree_node.tree.root
        if tree_node == root:
            for child in tree_node.children:
                self.update_source_tree_selections(child, theme)
            return
        assert tree_node.data is not None
        tree_node.data.decendant_included = False
        tree_node.data.decendant_error = False
        for child in tree_node.children:
            self.update_source_tree_selections(child, theme)
            child_data = child.data
            assert child_data is not None
            if child_data.included:
                tree_node.data.decendant_included = True
                if child_data.has_error():
                    tree_node.data.decendant_error = True
        tree_node.label = tree_node.data.get_text(theme)

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
                data = SourceTreeData.new_leaf_data(
                    f"{source_node.get_name()}", this_path
                )
            else:
                data = SourceTreeData.new_directory_data(f"{source_node.get_name()}")
            new_tree_node = dest_tree_node.add_leaf(
                f"{source_node.get_name()}", data=data
            )
            self.append_nodes(new_tree_node, source_node, this_path)
