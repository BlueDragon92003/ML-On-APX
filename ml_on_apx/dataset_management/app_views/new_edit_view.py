import re
from typing import Literal
from ml_on_apx.dataset_management.app_views.list_select_question import (
    ListSelectQuestion,
)
from ml_on_apx.dataset_management.app_views.binary_modal_question import (
    BinaryModalQuestion,
)
from ml_on_apx.dataset_management.app_views.source_tree_widget import (
    SourceTreeData,
    SourceTreeWidget,
)
from textual import on
from textual.widgets.tree import TreeNode, NodeID
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
from ml_on_apx.labelling import Label as SourceLabel, Labels
from textual.reactive import reactive
from ml_on_apx.dataset_management.dataset_info import DatasetInfo
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from ml_on_apx.dataset_management.tree import TreeNode as SourceTreeNode
from textual.screen import Screen


class NewEditView(Screen[None]):
    BINDINGS = [
        ("ctrl+b", "to_basic_info", "Basic info"),
        ("ctrl+l", "to_labels", "Labels"),
        ("ctrl+s", "to_sources", "Sources"),
        ("backspace", "delete_label", "Delete selected label"),
        Binding("ctrl+delete", "force_delete_label", show=False),  # without asking
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
        self.template_sources = {}
        self._highlighted_node: None | NodeID = None
        if template is not None:
            for label in template.get_labels():
                self.labels = self.labels + [label]
            for path, label in template.get_labeled_sources():
                self.template_sources.update({path: label})

    def compose(self) -> ComposeResult:
        yield Header(
            name="New Dataset" if self.dataset_name else f"Edit {self.dataset_name}"
        )
        yield Footer()
        with TabbedContent(id="new-edit-tabs"):
            with TabPane("Basic Info", id="general-info-tab"):
                with VerticalScroll():
                    with HorizontalGroup():
                        yield Label("Name:", id="name-label")
                        yield Input(
                            value=self.dataset_name,
                            placeholder="Dataset name...",
                            id="dataset-name-input",
                        )
                    with HorizontalGroup(id="new-edit-control-buttons"):
                        yield Button(
                            "Cancel", variant="default", id="dataset-cancel-button"
                        )
                        yield Button(
                            "Create", variant="primary", id="dataset-create-button"
                        )
            with TabPane("Labels", id="labels-tab"):
                yield ListView(id="labels-list")
                with HorizontalGroup(id="new-label"):
                    yield Input(
                        placeholder="Label name...",
                        id="label-name-input",
                    )
                    yield Button(
                        "New Label", variant="primary", id="label-create-button"
                    )
            with TabPane("Sources", id="sources-tab"):
                self._tree = SourceTreeWidget(
                    "/".join(self._manager.get_root_dir_path().parts) + "/",
                    id="source-tree",
                )
                self._tree.auto_expand = False
                yield self._tree

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        tabs = self.get_child_by_id("new-edit-tabs")
        assert type(tabs) is TabbedContent
        match tabs.active:
            case "general-info-tab":
                match action:
                    case "to_basic_info":
                        return None
                    case "to_labels":
                        return True
                    case "to_sources":
                        return True
            case "labels-tab":
                match action:
                    case "to_basic_info":
                        return True
                    case "to_labels":
                        return None
                    case "to_sources":
                        return True
                    case "delete_label":
                        return True
                    case "force_delete_label":
                        return True
            case "sources-tab":
                match action:
                    case "to_basic_info":
                        return True
                    case "to_labels":
                        return True
                    case "to_sources":
                        return None
                    case "set_source_label":
                        return True
        return False

    def on_mount(self):
        self.append_nodes(
            self._tree.root,
            self._manager.get_sources(),
            Path(self._manager.ROOT_DIR_NAME),
        )
        self.remake_label_list()

    @on(Tree.NodeHighlighted)
    def track_highlighted(self, message: Tree.NodeHighlighted[SourceTreeData]):
        self._highlighted_node = message.node.id
        message.stop()

    @on(Tree.NodeSelected)
    def handle_source_selection(self, message: Tree.NodeSelected[SourceTreeData]):
        if message.node.data is None:
            message.stop()
            return
        if message.node.parent is None:
            message.stop()
            return
        node = message.node
        data = message.node.data

        # self.release_children(message.node.parent)
        # or child in message.node.children:
        #             self.disinclude(child)

        #
        match data.inclusion:
            case _inclusion if _inclusion == data.InclusionType.DIRECTLY_INCLUDED:

                def included_node(selected: SourceLabel | Literal[False] | None):
                    if not selected:
                        if selected is None:
                            return
                        else:
                            self.disinclude(node)
                    else:
                        data.set_label(selected)
                        for child in node.children:
                            self.include(child)
                    self.update_source_tree_selections(self._tree.root)

                options: list[tuple[str, SourceLabel | Literal[False]]] = list()
                for label in self.labels:
                    options.append((str(label), label))
                options.append((f"> Remove {data.get_name()} from the dataset.", False))

                self.app.push_screen(
                    ListSelectQuestion(
                        options,
                        title=f"Which label should be used for `{data.get_name()}`?",
                    ),
                    callback=included_node,
                )
            case _inclusion if _inclusion == data.InclusionType.ANCENSTOR_INCLUDED:

                def ancestrally_included_node(
                    selected: SourceLabel | Literal[False] | None,
                ):
                    if selected is None:
                        return
                    else:
                        assert node.parent is not None
                        self.release_children(node.parent)
                        if selected:
                            data.set_label(selected)
                            data.inclusion = data.InclusionType.DIRECTLY_INCLUDED
                            for child in node.children:
                                self.include(child)
                        else:
                            self.disinclude(node)
                    self.update_source_tree_selections(self._tree.root)

                options: list[tuple[str, SourceLabel | Literal[False]]] = list()
                for label in self.labels:
                    options.append((str(label), label))
                options.append((f"> Remove {data.get_name()} from the dataset.", False))

                self.app.push_screen(
                    ListSelectQuestion(
                        options,
                        title=f"Which label should be used for `{data.get_name()}`?",
                        subtitle="Press escape to cancel.",
                    ),
                    callback=ancestrally_included_node,
                )
            case _inclusion if _inclusion == data.InclusionType.NOT_INCLUDED:

                def not_included_node(selected: SourceLabel | None):
                    if not selected:
                        return
                    else:
                        data.set_label(selected)
                        data.inclusion = data.InclusionType.DIRECTLY_INCLUDED
                        for child in node.children:
                            self.include(child)
                    self.update_source_tree_selections(self._tree.root)

                options: list[tuple[str, SourceLabel]] = list()
                for label in self.labels:
                    options.append((str(label), label))

                self.app.push_screen(
                    ListSelectQuestion(
                        options,
                        title=f"Which label should be used for `{data.get_name()}`?",
                    ),
                    callback=not_included_node,
                )

        message.stop()

    @on(Button.Pressed)
    def handle_button_press(self, message: Button.Pressed):
        match message.button.id:
            case "label-create-button":
                self.create_label()
            case "dataset-cancel-button":
                self.dismiss()
            case "dataset-create-button":
                self.validate()

    @on(Input.Submitted)
    def handle_input_submission(self, message: Input.Submitted):
        match message.input.id:
            case "label-name-input":
                self.create_label()

    def action_to_basic_info(self):
        tabs = self.get_child_by_id("new-edit-tabs")
        assert type(tabs) is TabbedContent
        tabs.active = "general-info-tab"
        tabs.get_pane("general-info-tab").focus()

    def action_to_labels(self):
        tabs = self.get_child_by_id("new-edit-tabs")
        assert type(tabs) is TabbedContent
        tabs.active = "labels-tab"
        self.get_widget_by_id("labels-list").focus()

    def action_to_sources(self):
        tabs = self.get_child_by_id("new-edit-tabs")
        assert type(tabs) is TabbedContent
        tabs.active = "sources-tab"
        self.get_widget_by_id("source-tree").focus()

    def action_delete_label(self):
        labels_list = self.get_widget_by_id("labels-list")
        assert type(labels_list) is ListView
        if labels_list.highlighted_child is not None:
            name = labels_list.highlighted_child.name

            def delete_label(delete: bool | None):
                if not delete:
                    return
                assert name is not None
                idx = self.labels.index(SourceLabel(name))
                self.labels = self.labels[:idx] + self.labels[idx + 1 :]

            self.app.push_screen(
                BinaryModalQuestion(Label(f"Delete label `{name}`?")), delete_label
            )

    def action_force_delete_label(self):
        labels_list = self.get_widget_by_id("labels-list")
        assert type(labels_list) is ListView
        if labels_list.highlighted_child is not None:
            name = labels_list.highlighted_child.name
            assert name is not None
            idx = self.labels.index(SourceLabel(name))
            self.labels = self.labels[:idx] + self.labels[idx + 1 :]

    def watch_labels(
        self, old_labels: list[SourceLabel], new_labels: list[SourceLabel]
    ):
        self.remake_label_list()
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

    def remake_label_list(self):
        labels_list = self.get_widget_by_id("labels-list")
        assert type(labels_list) is ListView
        labels_list.clear()
        for label in self.labels:
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
            if (
                self.template_sources is not None
                and this_path in self.template_sources.keys()
            ):
                assert new_tree_node.data is not None
                new_tree_node.data.inclusion = (
                    new_tree_node.data.InclusionType.DIRECTLY_INCLUDED
                )
                new_tree_node.data.set_label(self.template_sources[this_path])
            self.append_nodes(new_tree_node, source_node, this_path)
        dest_tree_node.expand()

    def release_children(self, node: TreeNode[SourceTreeData]):
        """Inform root ancestor to include its children, not itself directly"""
        assert node.data is not None
        if node.data.inclusion == node.data.InclusionType.ANCENSTOR_INCLUDED:
            assert node.parent is not None
            self.release_children(node.parent)
            node.data.inclusion = node.data.InclusionType.NOT_INCLUDED
            node.data.set_label(None)
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

    def create_label(self):
        input = self.get_widget_by_id("label-name-input")
        assert type(input) is Input
        label_name = input.value
        if not label_name:
            self.app.notify("Please input a label name.")
            input.focus()
            return
        label_name = re.sub(r"\s+", "-", label_name.lower())
        if not re.fullmatch(r"[\w-]+", label_name):
            self.app.notify("Label name is invalid.", severity="error")
            input.focus()
            return
        label = SourceLabel(label_name)
        if label in self.labels:
            self.app.notify(f"Label `{label_name}` already exists.", severity="error")
            input.focus()
            return
        # self.labels.append(label)
        self.labels = self.labels + [label]
        # self.watch_labels()
        input.clear()
        input.focus()

    def validate(self):
        name_wdgt = self.get_widget_by_id("dataset-name-input")
        assert type(name_wdgt) is Input
        name = name_wdgt.value
        labels = Labels(self.labels)

        try:
            labeled_sources: list[tuple[Path, SourceLabel]] = list(
                self._tree.get_labeled_sources()
            )
        except SourceTreeWidget.MissingLabelExeption as mle:
            self.app.notify(f"Source at {mle.path} has no label.", severity="error")
            return

        name = re.sub(r"\s+", "-", name)
        if not re.fullmatch(r"([\w-][\w\s-]*)?[\w-]", name):
            self.app.notify("Invalid dataset name.", severity="error")
            return

        dataset_info = DatasetInfo(labels, labeled_sources)
        self._manager.create_dataset(name, dataset_info)
        self.dismiss()
