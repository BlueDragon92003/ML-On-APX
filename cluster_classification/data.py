import os
import pathlib
import pickle
import textwrap
from enum import Enum

from torch.utils.data import DataLoader

from cluster_classification.cluster_classification_dataset import (
    ClusterClassificationDataset,
)
from cluster_classification.dataset_subset import DatasetSubset
from cluster_classification.classification_logger import CleverLogger

logger = CleverLogger(__name__)

logger.log_start_load_module()

class DatasourceType(Enum):
    """Marks if the datasource is for training or testing purposes"""

    TRAINING = 1
    TESTING = 2


def get_data(
    datasource_type: DatasourceType,
    batch_size: int,
) -> DataLoader:
    """Return a PyTorch DataLoader with a specified batch size and datatype.

    Arguments:
    - `datasource_type`: If the data is for testing or training purposes.
    - `batch_size`: the size of batches the dataloader provides.

    Returns:
    - A PyTorch dataloader that serves cluster classification data.
    """
    logger.log_enter_function('get_data', datasource_type=datasource_type, batch_size=batch_size)

    data = load_data(datasource_type, DatasetSubset.DOUBLE_ELECTRON)
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

    logger.log_function_exit_type('return', retval=data_loader)
    logger.log_exit_function('get_data')
    return data_loader


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
    logger.log_enter_function('load_data', datasource_type=datasource_type, datasets=datasets)

    # Extract the filepath where a picked file is/will be stored
    dataset_id = datasets.get_hex()
    
    module_path = pathlib.Path(__file__).parent
    project_path = module_path.parent
    data_path = project_path / 'data'
    root_data_path = data_path / 'classification'
    
    pickle_path = data_path / 'pickled' / 'classification' / \
        ('/'.join(textwrap.wrap(dataset_id, 4)) + '.pckl')
    
    logger.log_variables(dataset_id=dataset_id, pickle_path=pickle_path)

    create_new = False

    # Check if the file needs to be created
    logger.log_open_control_flow('pickle_exists_if_statement')
    if os.path.isfile(pickle_path):
        logger.log_control_element('ThenBranch')
        # Pickled file exists, check if it's outdated:
        m_time = os.path.getmtime(pickle_path)
        ccd_path = module_path / 'cluster_classification_dataset.py'

        # Check if it's an old version of the class
        logger.log_open_control_flow('outdated_fmt_if_statement')
        if os.path.getmtime(ccd_path) > m_time:
            logger.log_control_element('ThenBranch')
            logger.log_notice('Outdated pickle format', pickle=pickle_path)
            create_new = True
        logger.log_close_control_flow('outdated_if_statement')
        
        # Check if it's outdated compared to its source datasets
        logger.log_open_control_flow('outdated_data_for_loop')
        for (component_filename, _) in datasets.get_data():
            logger.log_control_element('Iteration', component_filename=component_filename)
            component_path = root_data_path / component_filename
            logger.log_open_control_flow('outdated_data_if_statement')
            if os.path.getmtime(component_path) > m_time:
                logger.log_control_element('ThenBranch')
                logger.log_notice("Outdated pickle data", pickle=pickle_path)
                create_new = True
            logger.log_close_control_flow('outdated_data_if_statement')
        logger.log_close_control_flow('outdated_data_for_loop')
    else:
        logger.log_control_element('ElseBranch')
        logger.log_notice("No pickle")
        create_new = True
    logger.log_close_control_flow('pickle_exists_if_statement')

    # Pickled file does not exist or is outdated; create a new one
    logger.log_open_control_flow('create_new_if_statement')
    if create_new:
        logger.log_control_element('ThenBranch')
        logger.log_start_minor_process("create_ccd_pickle")
        components = set()
        logger.log_open_control_flow('components_for_loop')
        for (component_filename, cluster_type) in datasets.get_data():
            logger.log_control_element('Iteration', component_filename=component_filename)
            component_path = root_data_path / component_filename
            components.add( (component_path, cluster_type) )
        logger.log_close_control_flow("components_for_loop")
        logger.log_variables(components=components)
        
        ccd = ClusterClassificationDataset(components)
        logger.log_variables(ccd=ccd)
        
        # ensure all directories are made
        os.makedirs(pickle_path.parent, exist_ok=True)
        pickle_fd = os.open(pickle_path, os.O_RDWR | os.O_CREAT ) 
        logger.log_open_control_flow('with_os_fdopen')
        with os.fdopen(pickle_fd, mode='wb') as pickled:
            pickle.dump(ccd, pickled)
        logger.log_close_control_flow('with_os_fdopen')
        logger.log_end_minor_process("create_ccd_pickle")
    # Pickled file exisits and is ready to use; load it
    else:
        logger.log_start_minor_process("load_ccd_pickle")
        pickle_fd = os.open(pickle_path, os.O_RDONLY)
        logger.log_open_control_flow('with_os_fdopen')
        with os.fdopen(pickle_fd, mode="rb") as pickled:
            ccd = pickle.load(pickled)
        logger.log_close_control_flow('with_os_fdopen')
        logger.log_end_minor_process('load_ccd_pickle')
    logger.log_close_control_flow('create_new_if_statement')

    logger.log_function_exit_type('return', retval=ccd)
    logger.log_exit_function('load_data')
    return ccd

logger.log_end_load_module()
