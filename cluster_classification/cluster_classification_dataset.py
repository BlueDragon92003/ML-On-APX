import math
from collections.abc import Callable
from typing import Set, Tuple, List, Union, Iterator

import numpy as np
import uproot
from torch.utils.data import IterableDataset, get_worker_info

from cluster_classification.classification_logger import CleverLogger
from cluster_classification.signal_type import SignalType

logger = CleverLogger(__name__)

logger.log_start_load_module()

class ClusterClassificationDataset(IterableDataset):
    """Implements a PyTorch dataset to train the cluster classifier.

    Extends: `torch.utils.data.IterableDataset`
    """

    def __init__(self, components: Set[Tuple[str, SignalType]]):
        """Generate a CCD using the provided file names and signal types.

        Arguments:
        - `components`: A set of (str, SignaTypes) that specify the paths to the
                        `.root` files this CCD reads from, and the signal type
                        each root file is associated with.
        """
        super(ClusterClassificationDataset, self).__init__()
        logger.log_enter_function('ccd_init', components = components)
        data = []
        # Data structure: data [cluster][features]
        logger.log_open_control_flow('init_for_loop')
        for filepath, cluster_type in components:
            logger.log_control_element('Iteration', filepath=filepath, cluster_type=cluster_type)
            logger.log_open_control_flow('open_uproot_with')
            with uproot.open(filepath) as file:
                tree = file["l1NtupleProducer/linkTree;1"]
                data.append(
                    np.fromiter(
                        cluster_generator(
                            lambda feature, event, card, final: tree[feature].array()[
                                event
                            ][card][final],
                            cluster_type,
                            num_events=len(tree["SLR0_cluster_eta"].array()),
                        ),
                        (int, 15),
                    )
                )
            logger.log_close_control_flow('open_uproot_with')
        logger.log_close_control_flow('init_for_loop')
        # # General data structure:
        # # data [ SLR/link + feature ][ event ][ card ]
        """
            0: cluster_et
            1: cluster_seed
            2: cluster_et5x5
            3: cluster_et2x5
            4: cluster_timing
            5: cluster_spike
            6: cluster_brems
            7: cluster_satur
            8: cluster_spare
            9: ecal_tower_et
        10: ecal_tower_timing
        11: ecal_tower_spike
        12: hcal_et
        13: hcal_fb
        14: classification
        """
        np_data = np.array(data)
        self.__data = np_data.reshape(-1, np_data.shape[-1])
        logger.log_exit_function('ccd_init')

    def __getitem__(self, index: int):
        out = self.__data[index]
        logger.log_micro_function('ccd_get_item', 'return', retval=out)
        return out

    def __len__(self):
        out = len(self.__data)
        logger.log_micro_function('ccd_len', 'return', retval=out)
        return out

    def __iter__(self):
        # Very much yoinked and adapted from the torch docs:
        # https://docs.pytorch.org/docs/2.9/data.html#torch.utils.data.Dataset

        logger.log_enter_function('ccd_iterator')
        worker_info = get_worker_info()
        logger.log_open_control_flow('worker_info_if_statement')
        if worker_info is None:
            logger.log_control_element('ThenBranch')
            # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self)
        else:  # in a worker process
            logger.log_control_element('ElseBranch')
            max_size = len(self)
            # split workload
            per_worker = int(
                math.ceil(len(self) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, max_size)
            logger.log_variables(iter_start=iter_start, iter_end=iter_end)
        retval=iter(self.__data[iter_start:iter_end])
        logger.log_function_exit_type('return', retval=retval)
        logger.log_exit_function('ccd_iterator')
        return retval


def get_ecal_tower(slr: int, i_eta: int, i_phi: int) -> int:
    """Calculates the index needed to extract the proper ECALUnclustered data.

    Arguments:
    - `slr`: The SLR on the RCT card being accessed.
    - `i_eta`: The rotational position accessed, in towers, in RCT coordinates.
    - `i_phi`: The horizontal position accessed, in towers, in RCT coordinates.

    `i_eta` is correlated with slr and all inputs must respect these brackets.
    This will be changed in the future.
    """
    logger.log_enter_function('get_ecal_tower', slr=slr, i_eta=i_eta, i_phi=i_phi)
    tower = 6 * i_eta + i_phi
    match slr:
        case 0:
            pass
        case 1:
            tower = tower - 12  # Index 0 in the SLR2 branch is eta=2,
            #   shifting indices by 12
        case 2:
            tower = tower - 42  # 12+30
        case 3:
            tower = tower - 72  # i_eta = 16, i_phi = 5 -> index 101
            #   101 - 72 = 29, the last index in SLR3 :)
    
    logger.log_function_exit_type('return', retval=tower)
    logger.log_exit_function('get_ecal_tower')
    return tower


def get_hcal_location(
    card: int, i_eta: int, i_phi: int
) -> Union[Tuple[int, int], None]:
    """Calculates the index to extract the proper HCAL data.

    Arguments:
    - `card`: The RCT card number being accessed.
    - `i_eta`: The rotational position accessed, in towers, in RCT coordinates.
    - `i_phi`: The horizontal position accessed, in towers, in RCT coordinates.
    """
    logger.log_enter_function('get_hcal_location', card=card, i_eta=i_eta, i_phi=i_phi)
    link = 5
    logger.log_open_control_flow('hcal_link_if_ladder')
    if i_eta > 15:
        logger.log_control_element('i_eta > 15', i_eta=i_eta)
        logger.log_close_control_flow('hcal_link_if_ladder')
        logger.log_function_exit_type('return', retval=None)
        logger.log_exit_function('get_hcal_location')
        return None
    elif i_eta > 7:
        logger.log_control_element('i_eta > 7', i_eta=i_eta)
        link += 1
        i_eta += -8  # index 0 of Link 6/8 is i_eta = 8
    logger.log_close_control_flow('hcal_link_if_ladder')

    high_link = i_phi + 6 * ((card % 4 - 1) // 2 % 2)
    # 1(True) if Links 5/6 start with 2, 0(False) otherwise
    # it's only ever actually used times 6 plus i_phi, so may as well do it here

    """
    | high_link | i_phi = 0 |   1   |   2   |   3   |   4   |   5   |
    | --------- | --------- | ----- | ----- | ----- | ----- | ----- |
    | False     | L5 i0     | L5 i1 | L5 i2 | L5 i3 | L7 i0 | L7 i1 |
    | True      | L7 i2     | L7 i3 | L5 i0 | L5 i1 | L5 i2 | L5 i3 |

    The formula `(i_phi + 6 * high_link) % 4` perfectly calculates the index (i)
    
    The formula `(i_phi + 6 * high_link) // 4 % 2 * 2` calculates the link
        offset (L-5)
    """

    tower_index = 4 * i_eta + high_link % 4
    link += high_link // 4 % 2 * 2

    logger.log_function_exit_type('return', retval=(tower_index, link))
    logger.log_exit_function('get_hcal_location')
    return tower_index, link


def cluster_generator(
    from_tree: Callable[[str, int, int, int], int],
    signal_type: SignalType,
    num_events: int,
) -> Iterator[List[int]]:
    """Generates a tuple of all datapoints corresponding to a cluster.

    Arguments:
    - `from_tree`: A function that accepts a feature, event number, card number,
                    and "final" number and pulls data from an attached source.
    - `signal_type`: The signal type of this dataset.
    - `num_events`: The number of events to pull from. Typically the number of
                    events in the dataset.
    """

    logger.log_enter_function('cluster_generator', from_tree=from_tree, signal_type=signal_type, num_events=num_events)
    logger.log_open_control_flow('cluster_for_loops')
    for event in range(num_events):
        for card in range(24):
            for slr in range(4):
                for cluster in range(9):
                    logger.log_control_element('Iteration', event=event, card=card, slr=slr, cluster=cluster)
                    # cluster i_eta and i_phi are in crystal;
                    # everyhting else is in tower
                    i_eta = (
                        from_tree(f"SLR{slr}_cluster_eta", event, card, cluster) // 5
                    )
                    i_phi = (
                        from_tree(f"SLR{slr}_cluster_phi", event, card, cluster) // 5
                    )
                    cluster_et = from_tree(
                        f"SLR{slr}_cluster_energy", event, card, cluster
                    )
                    # Energy can be left in its integer format; the specific
                    # values don't matter so much as the scale

                    logger.log_open_control_flow('is_cluster_if_statement')
                    if cluster_et == 0:
                        logger.log_control_element('ThenBranch')
                        # Not a cluster
                        logger.log_close_control_flow('is_cluster_if_statement')
                        logger.log_control_element('Continue')
                        continue
                    logger.log_close_control_flow('is_cluster_if_statement')

                    ecal_tower = get_ecal_tower(slr, i_eta, i_phi)
                    hcal_info = get_hcal_location(card, i_eta, i_phi)
                    logger.log_variables(ecal_tower=ecal_tower, hcal_info=hcal_info)

                    hcal_et = -1
                    hcal_fb = -1
                    logger.log_open_control_flow('hcal_info_exists_if_statement')
                    if hcal_info:
                        # HCAL information exists
                        logger.log_control_element('ThenBracnh')
                        (hcal_tower, link) = hcal_info
                        hcal_et = from_tree(
                            f"HCAL{link}_tower_et", event, card, hcal_tower
                        )
                        hcal_fb = from_tree(
                            f"HCAL{link}_tower_fb", event, card, hcal_tower
                        )
                    logger.log_close_control_flow('hcal_info_exists_if_statement')
                    
                    logger.log_function_exit_type('yield')
                    logger.log_exit_function('cluster_generator')
                    yield [
                        cluster_et,
                        from_tree(
                            f"SLR{slr}_cluster_seed_energy", event, card, cluster
                        ),
                        from_tree(f"SLR{slr}_cluster_et5x5", event, card, cluster),
                        from_tree(f"SLR{slr}_cluster_et2x5", event, card, cluster),
                        from_tree(f"SLR{slr}_cluster_timing", event, card, cluster),
                        0,  # from_tree(
                        #     f'SLR{slr}_cluster_spike', event, card, cluster
                        #     ),
                        from_tree(f"SLR{slr}_cluster_brems", event, card, cluster),
                        0,  # from_tree(
                        #     f'SLR{slr}_cluster_satur', event, card, cluster
                        #     ),
                        from_tree(f"SLR{slr}_cluster_spare", event, card, cluster),
                        from_tree(
                            f"ECALUnclusteredSLR{slr}_tower_et", event, card, ecal_tower
                        ),
                        from_tree(
                            f"ECALUnclusteredSLR{slr}_tower_timing",
                            event,
                            card,
                            ecal_tower,
                        ),
                        from_tree(
                            f"ECALUnclusteredSLR{slr}_tower_spike",
                            event,
                            card,
                            ecal_tower,
                        ),
                        hcal_et,
                        hcal_fb,
                        int(signal_type),
                    ]
                    logger.log_enter_function('cluster_generator')
    logger.log_close_control_flow('cluster_for_loops')
    logger.log_exit_function('cluster_generator')


logger.log_end_load_module()
