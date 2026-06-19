from typing import Type
from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.modes import Mode
from pathlib import Path
from ml_on_apx.dataset_management.dataset_manager import DatasetManager
from textual.app import App


class DatasetManagerApp(App):
    def __init__(self, active_dataset_manager: DatasetManager):
        self._manager = active_dataset_manager


def main(dataset_dir: Path, mode: Mode, dataset_class: Type[Dataset]):
    with DatasetManager(dataset_dir, mode, dataset_class) as manager:
        app = DatasetManagerApp(manager)
        app.run()
        print("(Re)compiling dataset pickles...")
