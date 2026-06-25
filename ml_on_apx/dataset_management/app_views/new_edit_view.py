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

    # TODO On node selected:
    #   add it to source list
    #   update its internal data
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
                yield Tree[tuple[str, SourceLabel | None]](
                    "/".join(self._manager.get_root_dir_path().parts) + "/"
                )

    def on_mount(self):
        self.remake_label_list(self.labels)

    def validate_dataset_name(self, new_name: str) -> str:
        return new_name

    def watch_labels(self, new_labels: list[SourceLabel]):
        self.remake_label_list(new_labels)

    def remake_label_list(self, labels: list[SourceLabel]):
        labels_list = self.get_widget_by_id("labels-list")
        assert type(labels_list) is ListView
        for label in labels:
            labels_list.append(ListItem(Label(f"{label}"), name=f"{label}"))

    def update_source_tree_selections(self):
        # TODO recursive for each node:
        #   Change color of text (hidden if unselected, green & bold if selected, red & bold if selected without label)
        #   Change label to "name (label)"
        pass

    def append_nodes(
        self,
        dest_tree_node: TreeNode,
        source_tree: SourceTreeNode,
    ):
        for source_node in source_tree.get_children():
            new_tree_node = dest_tree_node.add_leaf(
                f"{source_node.get_name()}",
            )
            self.append_nodes(new_tree_node, source_node)
