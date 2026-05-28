import os
import pickle
import textwrap
from enum import Enum

from torch.utils.data import DataLoader

from cluster_classification.cluster_classification_dataset import (
    ClusterClassificationDataset,
)
from cluster_classification.dataset_subset import DatasetSubset
from cluster_classification.classification_logger import ClassificationLogger

logger = ClassificationLogger()


class DatasourceType(Enum):
    """Marks if the datasource is for training or testing purposes"""

    TRAINING = 1
    TESTING = 2


logger.log_debug("Loaded Datasouce types")


def get_data(
    datasource_type: DatasourceType,
    batch_size: int,
):
    """Return a PyTorch DataLoader with a specified batch size and datatype.

    Arguments:
    - `datasource_type`: If the data is for testing or training purposes.
    - `batch_size`: the size of batches the dataloader provides.

    Returns:
    - A PyTorch dataloader that serves cluster classification data.
    """
    logger.log_info("Loading data...")
    data = load_data(datasource_type, DatasetSubset.DOUBLE_ELECTRON)
    return DataLoader(
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
    # Extract the filepath where a picked file is/will be stored
    dataset_id = datasets.get_hex()
    pickle_path = (
        "./data/pickled/classification/"
        + "/".join(textwrap.wrap(dataset_id, 4))
        + ".pckl"
    )

    logger.log_debug(f"dataset ID: {dataset_id}")
    logger.log_debug(f"pickle path: {pickle_path}")

    create_new = False

    # Check if the file needs to be created
    if os.path.isfile(pickle_path):
        # Pickled file exists, check if it's outdated:
        m_time = os.path.getmtime(pickle_path)
        # Check if it's an old version of the class
        if os.path.getmtime("./cluster_classification_dataset.py") > m_time:
            logger.log_debug("Outdated pickle format")
            create_new = True
        # Check if it's outdated compared to its source datasets
        for component_filename, _ in datasets.get_data():
            component_path = "./data/classification/" + component_filename
            if os.path.getmtime(component_path) > m_time:
                logger.log_debug(f"Outdated pickle {component_filename}")
                create_new = True
    else:
        logger.log_debug("No pickle")
        create_new = True

    # Pickled file does not exist or is outdated; create a new one
    # TODO needs to make the directory to work :/
    if create_new:
        logger.log_info("Creating new CCD pickle...")
        components = set()
        for component_filename, cluster_type in datasets.get_data():
            logger.log_debug(f"Dataset includes file {component_filename}")
            component_path = "./data/classification/" + component_filename
            components.add((component_path, cluster_type))
        ccd = ClusterClassificationDataset(components)
        pickle_fd = os.open(pickle_path, os.O_RDWR | os.O_CREAT)
        with os.fdopen(pickle_fd, mode="wb") as pickled:
            pickle.dump(ccd, pickled)
    # Pickled file exisits and is ready to use; load it
    else:
        logger.log_info("Loading pickled CCD...")
        pickle_fd = os.open(pickle_path, os.O_RDONLY)
        with os.fdopen(pickle_fd, mode="rb") as pickled:
            ccd = pickle.load(pickled)

    return ccd
