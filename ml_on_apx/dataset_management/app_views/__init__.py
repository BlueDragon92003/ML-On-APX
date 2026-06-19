"""


MainView(Window)
RenameView(Popup, old_name: str)
    RenameViewDone(new_name: str)
ConfirmDialogueView(Popup; title: str, content: Widget)
    ConfirmDialogViewDone(clicked_yes: bool)
SetNameView(Popup)
    SetNameViewDone(name: str)
EditSourcesView(Window; sources: TreeNode, already_selected: set[str]|None)
    EditSourcesViewDone(selected_sources: TreeNode)
EditLabelsView(Window; already_there: Labels|None)
    EditLabelsViewDone(labels: Labels)
AssignLabelsView(Window; sources: TreeNode, labels: Labels, previous_labelling: Tuple[Labels,Set[Tuple[Path, int]]]|None)
    AssignLabelsViewDone(assignment: Set[Tuple[Path, int]])
"""
