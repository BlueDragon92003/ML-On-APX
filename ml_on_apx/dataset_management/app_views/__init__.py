"""Contains components used to produce the TUI for dataset management."""

from ml_on_apx.dataset_management import _TUI
from ml_on_apx.dataset_management.dataset_info import DatasetInfo
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from ml_on_apx.logging import log_call


@log_call(action_type="ds_markdown" @ _TUI)
def get_dataset_info_markdown(dsinfo: DatasetInfo, manager: DatasetManager) -> str:
    """Produce a markdown summary from a DatasetInfo object.

    Args:
        dsinfo (DatasetInfo): The DatasetInfo object the markdown should be produced
            from.
        manager (DatasetManager): The DatasetManager that owns `dsinfo`

    Returns:
        str: The markdown string for `dsinfo`.

    """
    markdown = "The dataset uses the following labels:\n"
    for label in dsinfo.get_labels():
        markdown += f"  - {label}\n"
    markdown += "\nThe dataset uses the following sources:\n"
    for source in dsinfo.get_labeled_sources():
        markdown += (
            f"  - {source[0].relative_to(manager.get_root_dir_path())} ({source[1]})\n"
        )
    return markdown
