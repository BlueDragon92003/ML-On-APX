from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from ml_on_apx.dataset_management.dataset_info import DatasetInfo


def get_dataset_info_markdown(dsinfo: DatasetInfo, manager: DatasetManager) -> str:
    markdown = "The dataset uses the following labels:\n"
    for label in dsinfo.get_labels():
        markdown += f"  - {label}\n"
    markdown += "\nThe dataset uses the following sources:\n"
    for source in dsinfo.get_labeled_sources():
        markdown += (
            f"  - {source[0].relative_to(manager.get_root_dir_path())} ({source[1]})\n"
        )
    return markdown
