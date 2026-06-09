import jaxtyping
from collections.abc import Callable
from typing import Set, Tuple, List, Iterator, Dict

import numpy as np
import uproot
import jax
import jax.numpy as jnp
from torch.utils.data import Dataset

from cleverlogger import CleverLogger
from cluster_classification.signal_type import SignalType

logger = CleverLogger(__name__)

logger.log_start_load_module()


class ClusterClassificationDataset(Dataset):
    """Implements a PyTorch dataset to train the cluster classifier.

    Extends: `torch.utils.data.IterableDataset`
    """

    labels: Dict[int, Tuple[SignalType, int]]

    def __init__(self, components: Set[Tuple[str, SignalType]]):
        """Generate a CCD using the provided file names and signal types.

        Arguments:
        - `components`: A set of (str, SignaTypes) that specify the paths to the
                        `.root` files this CCD reads from, and the signal type
                        each root file is associated with.
        """
        super(ClusterClassificationDataset, self).__init__()
        logger.log_enter_function("ccd_init", components=components)
        data = []

        signal_to_label, self.labels = get_labels(components)

        # Data structure: data [cluster][features]
        logger.log_preloop("init_for_loop")
        for filepath, cluster_type in components:
            logger.log_iteration_head(filepath=filepath, cluster_type=cluster_type)
            process_clusters = jax.vmap(process_cluster)
            with uproot.open(filepath) as file:
                tree = file["l1NtupleProducer/linkTree;1"]
                data = []
                localdata = []  # TODO remove
                for cluster in range(9):
                    # TODO how to turn the tree data into a mega-list of everything to parallelize?
                    data.append(process_clusters(tree["SLR0_cluster_energy"]))

                # increase size
                curr_len = self.labels[signal_to_label[cluster_type]][1]
                self.labels[signal_to_label[cluster_type]] = (
                    cluster_type,
                    curr_len + len(localdata),
                )
                data.append(localdata)
            logger.log_iteration_tail()
        logger.log_postloop("init_for_loop")
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
        logger.log_exit_function("ccd_init")

    def __getitem__(self, index: int):
        out = self.__data[index]
        logger.log_micro_function("ccd_get_item", "return", retval=out)
        return out

    def __len__(self):
        out = len(self.__data)
        logger.log_micro_function("ccd_len", "return", retval=out)
        return out


@jax.jit
def get_ecal_tower(slr: int, i_eta: int, i_phi: int) -> int:
    """Calculates the index needed to extract the proper ECALUnclustered data.

    Arguments:
    - `slr`: The SLR on the RCT card being accessed.
    - `i_eta`: The rotational position accessed, in towers, in RCT coordinates.
    - `i_phi`: The horizontal position accessed, in towers, in RCT coordinates.

    `i_eta` is correlated with slr and all inputs must respect these brackets.
    This will be changed in the future.
    """
    # logger.log_enter_function("get_ecal_tower", slr=slr, i_eta=i_eta, i_phi=i_phi)
    tower = 6 * i_eta + i_phi
    tower = jax.lax.switch(
        slr,
        [
            lambda tower: tower,
            lambda tower: tower - 12,
            lambda tower: tower - 42,
            lambda tower: tower - 72,
        ],
        tower,
    )
    # match slr:
    #     case 0:
    #         pass
    #     case 1:
    #         tower = tower - 12  # Index 0 in the SLR2 branch is eta=2,
    #         #   shifting indices by 12
    #     case 2:
    #         tower = tower - 42  # 12+30
    #     case 3:
    #         tower = tower - 72  # i_eta = 16, i_phi = 5 -> index 101
    #         #   101 - 72 = 29, the last index in SLR3 :)

    # logger.log_function_exit_type("return", retval=tower)
    # logger.log_exit_function("get_ecal_tower")
    return tower


@jax.jit
def get_hcal_location(card: int, i_eta: int, i_phi: int) -> Tuple[int, int]:
    """Calculates the index to extract the proper HCAL data.

    Arguments:
    - `card`: The RCT card number being accessed.
    - `i_eta`: The rotational position accessed, in towers, in RCT coordinates.
    - `i_phi`: The horizontal position accessed, in towers, in RCT coordinates.
    """
    # logger.log_enter_function("get_hcal_location", card=card, i_eta=i_eta, i_phi=i_phi)
    high_link = i_phi + 6 * ((card % 4 - 1) // 2 % 2)
    # 1(True) if Links 5/6 start with 2, 0(False) otherwise
    # it's only ever actually used times 6 plus i_phi, so may as well do it here

    return jax.lax.cond(
        i_eta > 15,  # cond
        (-1, -1),  # true -- no HCAL data for i_eta 16
        jax.lax.cond(  # false -- HCAL data may exist
            i_eta > 7,  # cond -- Link 5/7 or 6/8?
            (  # true -- link 6 or 8
                4 * (i_eta - 8) + high_link % 4,  # tower_index
                6 + (high_link // 4 % 2 * 2),  # link
            ),
            (  # false -- link 5 or 7
                4 * (i_eta) + high_link % 4,  # tower_index
                5 + (high_link // 4 % 2 * 2),  # link
            ),
        ),
    )
    high_link = i_phi + 6 * ((card % 4 - 1) // 2 % 2)

    """
    | high_link | i_phi = 0 |   1   |   2   |   3   |   4   |   5   |
    | --------- | --------- | ----- | ----- | ----- | ----- | ----- |
    | False     | L5 i0     | L5 i1 | L5 i2 | L5 i3 | L7 i0 | L7 i1 |
    | True      | L7 i2     | L7 i3 | L5 i0 | L5 i1 | L5 i2 | L5 i3 |

    The formula `(i_phi + 6 * high_link) % 4` perfectly calculates the index (i)
    
    The formula `(i_phi + 6 * high_link) // 4 % 2 * 2` calculates the link
        offset (L-5)
    """


@jax.jit
def process_cluster(
    slr: int,
    event: int,
    card: int,
    cluster_eta: int,
    cluster_phi: int,
    cluster_energy: int,
    cluster_seed_energy: int,
    cluster_et5x5: int,
    cluster_et2x5: int,
    cluster_timing: int,
    cluster_spike: int,
    cluster_brems: int,
    cluster_satur: int,
    cluster_spare: int,
    unclustered_ets: jaxtyping.Int[jax.Array, " etower"],  # noqa: F722
    unclustered_timings: jaxtyping.Int[jax.Array, " etower"],  # noqa: F722
    unclustered_spikes: jaxtyping.Int[jax.Array, " etower"],  # noqa: F722
    hcal5_ets: jaxtyping.Int[jax.Array, " htower"],  # noqa: F722
    hcal5_fbs: jaxtyping.Int[jax.Array, " htower"],  # noqa: F722
    hcal6_ets: jaxtyping.Int[jax.Array, " htower"],  # noqa: F722
    hcal6_fbs: jaxtyping.Int[jax.Array, " htower"],  # noqa: F722
    hcal7_ets: jaxtyping.Int[jax.Array, " htower"],  # noqa: F722
    hcal7_fbs: jaxtyping.Int[jax.Array, " htower"],  # noqa: F722
    hcal8_ets: jaxtyping.Int[jax.Array, " htower"],  # noqa: F722
    hcal8_fbs: jaxtyping.Int[jax.Array, " htower"],  # noqa: F722
) -> jax.Array:
    i_eta = cluster_eta // 5
    i_phi = cluster_phi // 5
    ecal_tower = get_ecal_tower(slr, i_eta, i_phi)
    hcal_tower, hcal_link = get_hcal_location(card, i_eta, i_phi)
    hcal_et = jax.lax.switch(
        hcal_link - 5,
        [
            lambda tower: hcal5_ets[tower],
            lambda tower: hcal6_ets[tower],
            lambda tower: hcal7_ets[tower],
            lambda tower: hcal8_ets[tower],
        ],
        hcal_tower,
    )
    hcal_fb = jax.lax.switch(
        hcal_link - 5,
        [
            lambda tower: hcal5_fbs[tower],
            lambda tower: hcal6_fbs[tower],
            lambda tower: hcal7_fbs[tower],
            lambda tower: hcal8_fbs[tower],
        ],
        hcal_tower,
    )
    return jnp.array(
        [
            slr,
            event,
            card,
            cluster_eta,
            cluster_phi,
            cluster_energy,
            cluster_seed_energy,
            cluster_et5x5,
            cluster_et2x5,
            cluster_timing,
            cluster_spike,
            cluster_brems,
            cluster_satur,
            cluster_spare,
            unclustered_ets[ecal_tower],
            unclustered_timings[ecal_tower],
            unclustered_spikes[ecal_tower],
            hcal_et,
            hcal_fb,
        ]
    )


def cluster_generator(
    from_tree: Callable[[str, int, int, int], int],
    label: int,
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

    logger.log_enter_function(
        "cluster_generator",
        from_tree=from_tree,
        label=label,
        num_events=num_events,
    )
    logger.log_preloop("cluster_for_loops")
    for event in range(num_events):
        for card in range(24):
            for slr in range(4):
                for cluster in range(9):
                    logger.log_iteration_head(
                        event=event, card=card, slr=slr, cluster=cluster
                    )
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

                    if cluster_et == 0:
                        # Not a cluster
                        logger.log_iteration_exit("continue")
                        logger.log_iteration_tail()
                        continue

                    ecal_tower = get_ecal_tower(slr, i_eta, i_phi)
                    hcal_info = get_hcal_location(card, i_eta, i_phi)
                    logger.log_variables(ecal_tower=ecal_tower, hcal_info=hcal_info)

                    hcal_et = -1
                    hcal_fb = -1
                    if hcal_info:
                        # HCAL information exists
                        (hcal_tower, link) = hcal_info
                        hcal_et = from_tree(
                            f"HCAL{link}_tower_et", event, card, hcal_tower
                        )
                        hcal_fb = from_tree(
                            f"HCAL{link}_tower_fb", event, card, hcal_tower
                        )

                    logger.log_iteration_exit("yield")
                    logger.log_iteration_tail()
                    logger.log_postloop("cluster_for_loops")
                    logger.log_function_exit_type("yield")
                    logger.log_exit_function("cluster_generator")
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
                        int(label),
                    ]
                    logger.log_enter_function("cluster_generator")
                    logger.log_preloop("cluster_for_loops")
    logger.log_postloop("cluster_for_loops")
    logger.log_exit_function("cluster_generator")


def get_labels(
    components: Set[Tuple[str, SignalType]],
) -> Tuple[Dict[SignalType, int], Dict[int, Tuple[SignalType, int]]]:
    """Create a set of contiguous integer labels for each signal type, with additional data storage."""
    logger.log_enter_function("get_lables", components=components)
    signal_to_label: Dict[SignalType, int] = dict()
    labels: Dict[int, Tuple[SignalType, int]] = dict()
    next_label = 0
    logger.log_preloop("labeling_for_loop")
    for _, signal_type in components:
        logger.log_iteration_head(signal_type=signal_type)
        if signal_type not in signal_to_label.keys():
            signal_to_label[signal_type] = next_label
            labels[next_label] = (signal_type, 0)
            next_label += 1
        logger.log_iteration_tail()
    logger.log_postloop("labeling_for_loop")
    logger.log_function_exit_type("return", retval=(signal_to_label, labels))
    logger.log_exit_function("get_labels")
    return signal_to_label, labels


logger.log_end_load_module()
