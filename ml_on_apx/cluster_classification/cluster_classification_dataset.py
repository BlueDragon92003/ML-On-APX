"""The dataset used by cluster classification models."""

from pathlib import Path

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import uproot
from uproot.behaviors.TBranch import TBranch

from ml_on_apx.cluster_classification import _CLASS
from ml_on_apx.dataset_management.dataset import Dataset
from ml_on_apx.logging import log_call

_CCD = "ds" @ _CLASS


class ClusterClassificationDataset(Dataset):
    """Implements a PyTorch dataset to train the cluster classifier.

    Extends: `torch.utils.data.IterableDataset`
    """

    size_per_label: dict[int, int]

    def __init__(self, components: set[tuple[Path, int]]) -> None:
        """Generate a CCD using the provided file names and ml labels.

        Arguments:
        components: A set of (str, SignaTypes) that specify the paths to the `.root`
            files this CCD reads from, and the signal type each root file is
            associated with.

        """
        super(ClusterClassificationDataset, self).__init__()
        data_parts = []
        self.size_per_label = {}

        # Data structure: data [cluster][features]
        for filepath, label in components:
            with uproot.open(filepath) as file:
                tree: uproot.ReadOnlyDirectory = file["l1NtupleProducer/linkTree;1"]
                localdata_parts = []
                num_events = len(tree["HCAL5_tower_et"].array(library="np"))
                hcal5_ets = mostly_flatten(tree["HCAL5_tower_et"]).repeat(9, axis=0)
                hcal5_fbs = mostly_flatten(tree["HCAL5_tower_fb"]).repeat(9, axis=0)
                hcal6_ets = mostly_flatten(tree["HCAL6_tower_et"]).repeat(9, axis=0)
                hcal6_fbs = mostly_flatten(tree["HCAL6_tower_fb"]).repeat(9, axis=0)
                hcal7_ets = mostly_flatten(tree["HCAL7_tower_et"]).repeat(9, axis=0)
                hcal7_fbs = mostly_flatten(tree["HCAL7_tower_fb"]).repeat(9, axis=0)
                hcal8_ets = mostly_flatten(tree["HCAL8_tower_et"]).repeat(9, axis=0)
                hcal8_fbs = mostly_flatten(tree["HCAL8_tower_fb"]).repeat(9, axis=0)
                for slr in range(4):
                    unclustered_ets = mostly_flatten(
                        tree[f"ECALUnclusteredSLR{slr}_tower_et"]
                    ).repeat(9, axis=0)
                    unclustered_timings = mostly_flatten(
                        tree[f"ECALUnclusteredSLR{slr}_tower_timing"]
                    ).repeat(9, axis=0)
                    unclustered_spikes = mostly_flatten(
                        tree[f"ECALUnclusteredSLR{slr}_tower_spike"]
                    ).repeat(9, axis=0)
                    localdata_parts.append(
                        process_clusters(
                            jnp.array([slr for _ in range(len(unclustered_ets))]),
                            jnp.arange(num_events).repeat(9 * 24),  # event
                            jnp.array(
                                [i for _ in range(num_events * 9) for i in range(24)]
                            ),  # card
                            mostly_flatten(tree[f"SLR{slr}_cluster_eta"]).ravel(),
                            mostly_flatten(tree[f"SLR{slr}_cluster_phi"]).ravel(),
                            mostly_flatten(tree[f"SLR{slr}_cluster_energy"]).ravel(),
                            mostly_flatten(
                                tree[f"SLR{slr}_cluster_seed_energy"]
                            ).ravel(),
                            mostly_flatten(tree[f"SLR{slr}_cluster_et5x5"]).ravel(),
                            mostly_flatten(tree[f"SLR{slr}_cluster_et2x5"]).ravel(),
                            mostly_flatten(tree[f"SLR{slr}_cluster_timing"]).ravel(),
                            jnp.zeros(9 * 24 * num_events),
                            # mostly_flatten(tree[f"SLR{slr}_cluster_spike"]).ravel(),
                            mostly_flatten(tree[f"SLR{slr}_cluster_brems"]).ravel(),
                            jnp.zeros(9 * 24 * num_events),
                            # mostly_flatten(tree[f"SLR{slr}_cluster_satur"]).ravel(),
                            # mostly_flatten(tree[f"SLR{slr}_cluster_spare"]).ravel(),
                            unclustered_ets,
                            unclustered_timings,
                            unclustered_spikes,
                            hcal5_ets,
                            hcal5_fbs,
                            hcal6_ets,
                            hcal6_fbs,
                            hcal7_ets,
                            hcal7_fbs,
                            hcal8_ets,
                            hcal8_fbs,
                            jnp.array([label for _ in range(num_events * 9 * 24)]),
                        )
                    )
                localdata = np.concatenate(
                    localdata_parts
                )  # sync up with jax execution
                localdata = localdata[localdata[:, 5] > 0]  # index 5 is cluster_energy
                # increase size
                curr_len = (
                    self.size_per_label[label]
                    if label in self.size_per_label.keys()
                    else 0
                )
                self.size_per_label.update({label: curr_len + len(localdata)})
                data_parts.append(localdata)
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
        data = np.concatenate(data_parts)
        self.__data = data  # .reshape(-1, data.shape[-1])

    def __getitem__(self, index: int) -> np.ndarray:
        """Return the item at index index.

        Arguments:
            index (int): The index to get from.

        Returns:
            np.ndarray : A 19-item array of integers.

        """
        out = self.__data[index]
        return out

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        out = len(self.__data)
        return out

    @classmethod
    @log_call(action_type="create" @ _CCD)
    def create(cls, components: set[tuple[Path, int]]) -> "Dataset":
        """Create a new ClusterClassificationDataset.

        Args:
            components (set[tuple[Path, int]]): A set of paths to ROOT files paired with
                integer labeles that the dataset will draw from.

        Returns:
            Dataset: The ClusterClassificationDataset built from the specified
            components.

        """
        return ClusterClassificationDataset(components)


def mostly_flatten(branch: TBranch) -> jax.Array:
    """Flatten all except the last dimension of a uproot TBranch.

    Args:
        branch (TBranch): _description_

    Returns:
        jax.Array: _description_

    """
    arr = branch.array()
    return jnp.array(np.array(arr.tolist()).reshape(-1, len(arr[0][0])))


@jax.jit
def get_ecal_tower(slr: int, i_eta: int, i_phi: int) -> int:
    """Calculate the index needed to extract the proper ECALUnclustered data.

    `i_eta` is correlated with slr and all inputs must respect these brackets.
    This will be changed in the future.

    Arguments:
        slr (int): The SLR on the RCT card being accessed.
        i_eta (int): The rotational position accessed, in towers, in RCT coordinates.
        i_phi (int): The horizontal position accessed, in towers, in RCT coordinates.

    Returns:
        int: The index where the specified position is stored.

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
def get_hcal_location(card: int, i_eta: int, i_phi: int) -> tuple[int, int]:
    """Calculate the index to extract the proper HCAL data.

    Arguments:
        card (int): The RCT card number being accessed.
        i_eta (int): The rotational position accessed, in towers, in RCT coordinates.
        i_phi (int): The horizontal position accessed, in towers, in RCT coordinates.

    Returns:
        tuple[int,int]: The HCAL link and index where the specified position is stored.

    """
    # logger.log_enter_function("get_hcal_location", card=card, i_eta=i_eta,
    # i_phi=i_phi)
    high_link = i_phi + 6 * ((card % 4 - 1) // 2 % 2)
    # 1(True) if Links 5/6 start with 2, 0(False) otherwise
    # it's only ever actually used times 6 plus i_phi, so may as well do it here

    return jax.lax.cond(
        i_eta > 15,  # cond
        lambda: (-1, -1),  # true -- no HCAL data for i_eta 16
        lambda: jax.lax.cond(  # false -- HCAL data may exist
            i_eta > 7,  # cond -- Link 5/7 or 6/8?
            lambda: (  # true -- link 6 or 8
                4 * (i_eta - 8) + high_link % 4,  # tower_index
                6 + (high_link // 4 % 2 * 2),  # link
            ),
            lambda: (  # false -- link 5 or 7
                4 * (i_eta) + high_link % 4,  # tower_index
                5 + (high_link // 4 % 2 * 2),  # link
            ),
        ),
    )

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
@jax.vmap
def process_clusters(
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
    # cluster_spare: int,
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
    label: int,
) -> jax.Array:
    """Process the possible cluster into a form useable by the trainer.

    Args:
        slr (int): The SLR number where the cluster is being processed.
        event (int): The event number the cluster occured in.
        card (int): The FPGA card where the cluster is being processed.
        cluster_eta (int): The card-relative ECAL crystal ieta position where the
            cluster is.
        cluster_phi (int): The card-relative ECAL crystal iphi position where the
            cluster is.
        cluster_energy (int): The total energy of the cluster.
        cluster_seed_energy (int): The seed energy of the cluster.
        cluster_et5x5 (int): The total energy of the cluster in a smaller are.
        cluster_et2x5 (int): The shaped energy of the cluster in a small area.
        cluster_timing (int): The timing information of the cluster.
        cluster_spike (int): The spike information of the cluster.
        cluster_brems (int): The Bremsstrahlung information of the cluster.
        cluster_satur (int): The saturation information of the cluster
        unclustered_ets (jaxtyping.Int): Array of total unclustered energies.
        unclustered_timings (jaxtyping.Int): Array of unclustered timing information.
        unclustered_spikes (jaxtyping.Int): Array of unclustered spike information.
        hcal5_ets (jaxtyping.Int): Array of HCAL energies in link 5.
        hcal5_fbs (jaxtyping.Int): Array of HCAL feature bits in link 5.
        hcal6_ets (jaxtyping.Int): Array of HCAL energies in link 6.
        hcal6_fbs (jaxtyping.Int): Array of HCAL feature bits in link 6.
        hcal7_ets (jaxtyping.Int): Array of HCAL energies in link 7.
        hcal7_fbs (jaxtyping.Int): Array of HCAL feature bits in link 7.
        hcal8_ets (jaxtyping.Int): Array of HCAL energies in link 8.
        hcal8_fbs (jaxtyping.Int): Array of HCAL feature bits in link 8.
        label (int): The label this cluster will be marked with for training.

    Returns:
        jax.Array: A 14-item array that includes the cluster data and its label.

    """
    i_eta = cluster_eta // 5
    i_phi = cluster_phi // 5
    ecal_tower = get_ecal_tower(slr, i_eta, i_phi)
    hcal_tower, hcal_link = get_hcal_location(card, i_eta, i_phi)

    hcal_et = jax.lax.cond(
        hcal_tower >= 0,
        lambda: jax.lax.switch(
            hcal_link - 5,
            [
                lambda tower: hcal5_ets[tower],
                lambda tower: hcal6_ets[tower],
                lambda tower: hcal7_ets[tower],
                lambda tower: hcal8_ets[tower],
            ],
            hcal_tower,
        ),
        lambda: -1,
    )
    hcal_fb = jax.lax.cond(
        hcal_tower >= 0,
        lambda: jax.lax.switch(
            hcal_link - 5,
            [
                lambda tower: hcal5_fbs[tower],
                lambda tower: hcal6_fbs[tower],
                lambda tower: hcal7_fbs[tower],
                lambda tower: hcal8_fbs[tower],
            ],
            hcal_tower,
        ),
        lambda: -1,
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
            # cluster_spare,
            unclustered_ets[ecal_tower],
            unclustered_timings[ecal_tower],
            unclustered_spikes[ecal_tower],
            hcal_et,
            hcal_fb,
            label,
        ]
    )
