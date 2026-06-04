from typing import Tuple
import os
import pathlib
import pickle
import textwrap
from enum import Enum

from torch import Tensor, log
from torch.nn.functional import normalize
from torch.utils.data import DataLoader

from cluster_classification.cluster_classification_dataset import (
    ClusterClassificationDataset,
)
from cleverlogger import CleverLogger
from cluster_classification.dataset_subset import DatasetSubset

logger = CleverLogger(__name__)

logger.log_start_load_module()


class DatasourceType(Enum):
    """Marks if the datasource is for training or testing purposes"""

    TRAINING = 1
    TESTING = 2


def get_data(
    datasource_type: DatasourceType,
    datasets: DatasetSubset,
    batch_size: int,
) -> Tuple[DataLoader, Tensor]:
    """Return a PyTorch DataLoader with a specified batch size and datatype.

    Arguments:
    - `datasource_type`: If the data is for testing or training purposes.
    - `batch_size`: the size of batches the dataloader provides.

    Returns:
    - A PyTorch dataloader that serves cluster classification data.
    """
    logger.log_enter_function(
        "get_data",
        datasource_type=datasource_type,
        datasets=datasets,
        batch_size=batch_size,
    )

    data = load_data(datasource_type, datasets)

    weights = [0 for _ in data.labels.keys()]
    for key in data.labels.keys():
        weights[key] = data.labels[key][1]
    weights = normalize(-1 * log(normalize(Tensor(weights), dim=0)))

    data_loader = DataLoader(
        # The data to load
        data,
        # Set the batch size
        batch_size=batch_size,
        # Provide the data in a random order. This improves training by
        #   giving the optimizer a more hollistic overview of the data,
        #   instead of unintentionally training & optimizing the model on
        #   one class, then the next class, then the next one, etc.
        shuffle=True,
    )

    logger.log_function_exit_type("return", retval=data_loader)
    logger.log_exit_function("get_data")
    return data_loader, weights


def load_data(
    datasource_type: DatasourceType, datasets: DatasetSubset
) -> ClusterClassificationDataset:
    """Intellegently loads or creates a `Dataset` given `DatasetSubsets`

    Loads the data as a PyTorch Dataset, provided a specific type (training or
    testing) and a set of ROOT data to use as samples.

    First, the function checks if such a Dataset has already been created. If
    so, it checks if the preserved Dataset needs to be updated (i.e. at least
    one of the datafiles is newer than the picked Dataset). If the preserved
    Dataset exists and is not outdated, it is unpicked and returned.

    If the preserved dataset is outdated or does not exist, then a new one is
    created from ROOT data.

    Arguments:
    - `datasource_type`: If the data is for training or testing purposes.
    - `datasets`: A dataset subset composed of varous `.root` files read for
                    data.

    Returns:
    - A PyTorch Dataset of the requested ROOT data.
    """
    logger.log_enter_function(
        "load_data", datasource_type=datasource_type, datasets=datasets
    )

    # Extract the filepath where a picked file is/will be stored
    dataset_id = datasets.get_hex()

    module_path = pathlib.Path(__file__).parent
    project_path = module_path.parent
    data_path = project_path / "data"
    root_data_path = data_path / "classification"

    pickle_path = (
        data_path
        / "pickled"
        / "classification"
        / ("/".join(textwrap.wrap(dataset_id, 4)) + ".pckl")
    )

    logger.log_variables(dataset_id=dataset_id, pickle_path=pickle_path)

    create_new = False

    # Check if the file needs to be created
    if os.path.isfile(pickle_path):
        # Pickled file exists, check if it's outdated:
        m_time = os.path.getmtime(pickle_path)
        ccd_path = module_path / "cluster_classification_dataset.py"

        # Check if it's an old version of the class
        if os.path.getmtime(ccd_path) > m_time:
            logger.log_notice("Outdated pickle format", pickle=pickle_path)
            create_new = True

        # Check if it's outdated compared to its source datasets
        logger.log_preloop("outdated_data_for_loop")
        for component_filename, _ in datasets.get_data():
            logger.log_iteration_head(component_filename=component_filename)
            component_path = root_data_path / component_filename
            if os.path.getmtime(component_path) > m_time:
                logger.log_notice("Outdated pickle data", pickle=pickle_path)
                create_new = True
            logger.log_iteration_tail()
        logger.log_postloop("outdated_data_for_loop")
    else:
        logger.log_notice("No pickle")
        create_new = True

    # Pickled file does not exist or is outdated; create a new one
    if create_new:
        logger.log_start_minor_process("create_ccd_pickle")
        components = set()
        logger.log_preloop("components_for_loop")
        for component_filename, cluster_type in datasets.get_data():
            logger.log_iteration_head(component_filename=component_filename)
            component_path = root_data_path / component_filename
            components.add((component_path, cluster_type))
            logger.log_iteration_tail()
        logger.log_postloop("components_for_loop")
        logger.log_variables(components=components)

        ccd = ClusterClassificationDataset(components)
        logger.log_variables(ccd=ccd)

        # ensure all directories are made
        os.makedirs(pickle_path.parent, exist_ok=True)
        pickle_fd = os.open(pickle_path, os.O_RDWR | os.O_CREAT)
        with os.fdopen(pickle_fd, mode="wb") as pickled:
            pickle.dump(ccd, pickled)
        logger.log_end_minor_process("create_ccd_pickle")
    # Pickled file exisits and is ready to use; load it
    else:
        logger.log_start_minor_process("load_ccd_pickle")
        pickle_fd = os.open(pickle_path, os.O_RDONLY)
        with os.fdopen(pickle_fd, mode="rb") as pickled:
            ccd = pickle.load(pickled)
        logger.log_end_minor_process("load_ccd_pickle")

    logger.log_function_exit_type("return", retval=ccd)
    logger.log_exit_function("load_data")
    return ccd


logger.log_end_load_module()
