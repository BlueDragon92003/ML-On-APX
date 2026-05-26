from typing import Set, Tuple
import math

from torch.utils.data import IterableDataset, get_worker_info
import uproot
import numpy as np

from signal_type import SignalType

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
        super(ClusterClassificationDataset).__init__()
        self.__data = []
        # Data structure: data [cluster][features]
        for (filepath, cluster_type) in components:
            with uproot.open(filepath) as file:
                tree = file["l1NtupleProducer/linkTree;1"]
                self.__data.append( np.fromiter(
                    cluster_generator(
                        lambda feature, event, card, final: \
                            tree[feature].array()[event][card][final],
                        cluster_type,
                        num_events=len(tree['SLR0_cluster_eta'].array())
                        )
                    (int,15)
                ) )               
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
        self.__data = self.__data.reshape(-1, self.__data.shape[-1])

    def __getitem__(self, index):
        return self.__data[index]

    def __len__(self):
        return len(self.__data)

    def __iter__(self):
        # Very much yoinked and adapted from the torch docs:
        # https://docs.pytorch.org/docs/2.9/data.html#torch.utils.data.Dataset

        worker_info = get_worker_info()
        if worker_info is None:
            # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(self)
        else:  # in a worker process
            max_size = len(self)
            # split workload
            per_worker = int(math.ceil((self.end - self.start) \
                / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, max_size)
        return iter(self.__data[iter_start:iter_end])

def get_ecal_tower(slr, i_eta, i_phi):
    """Calculates the index needed to extract the proper ECALUnclustered data.

    Arguments:
    - `slr`: The SLR on the RCT card being accessed.
    - `i_eta`: The rotational position accessed, in towers, in RCT coordinates.
    - `i_phi`: The horizontal position accessed, in towers, in RCT coordinates.

    `i_eta` is correlated with slr and all inputs must respect these brackets.
    This will be changed in the future.
    """
    tower = 6*i_eta + i_phi 
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
    return tower

def get_hcal_location(card, i_eta, i_phi):
    """Calculates the index to extract the proper HCAL data.

    Arguments:
    - `card`: The RCT card number being accessed.
    - `i_eta`: The rotational position accessed, in towers, in RCT coordinates.
    - `i_phi`: The horizontal position accessed, in towers, in RCT coordinates.
    """
    link = 5
    if i_eta > 15:
        return None
    elif i_eta > 7:
        link += 1
        i_eta += -8 # index 0 of Link 6/8 is i_eta = 8

    high_link = i_phi + 6 * ( (card % 4 - 1) // 2 % 2 ) 
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

    tower_index = 4*i_eta + high_link % 4
    link += high_link // 4 % 2 * 2

    return tower_index, int(link)

def cluster_generator(from_tree, signal_type, num_events):
    """Generates a tuple of all datapoints corresponding to a cluster.

    Arguments:
    - `from_tree`: A function that accepts a feature, event number, card number,
                    and "final" number and pulls data from an attached source. 
    - `signal_type`: The signal type of this dataset.
    - `num_events`: The number of events to pull from. Typically the number of
                    events in the dataset.
    """
    
    for event in range(num_events):
        for card in range(24):
            for slr in range(4):
                for cluster in range(9):
                    # cluster i_eta and i_phi are in crystal;
                    # everyhting else is in tower
                    i_eta = from_tree(
                        f'SLR{slr}_cluster_eta', event, card, cluster
                        ) // 5
                    i_phi = from_tree(
                        f'SLR{slr}_cluster_phi', event, card, cluster
                        ) // 5
                    cluster_et = from_tree(
                        f'SLR{slr}_cluster_energy', event, card, cluster
                        )
                    # Energy can be left in its integer format; the specific
                    # values don't matter so much as the scale

                    if cluster_et == 0:
                        # Not a cluster
                        continue

                    ecal_tower = get_ecal_tower(slr, i_eta, i_phi)
                    hcal_info = get_hcal_location(card, i_eta, i_phi)
                    
                    hcal_et = -1
                    hcal_fb = -1
                    if hcal_info:
                        # HCAL information exists
                        (hcal_tower, link) = hcal_info
                        hcal_et = from_tree(
                            f'HCAL{link}_tower_et', event, card, hcal_tower
                            )
                        hcal_fb = from_tree(
                            f'HCAL{link}_tower_fb', event, card, hcal_tower
                            )
                    
                    yield [
                        cluster_et,
                        from_tree(
                            f'SLR{slr}_cluster_seed_energy',
                            event, card, cluster
                            ),
                        from_tree(
                            f'SLR{slr}_cluster_et5x5', event, card, cluster
                            ),
                        from_tree(
                            f'SLR{slr}_cluster_et2x5', event, card, cluster
                            ),
                        from_tree(
                            f'SLR{slr}_cluster_timing', event, card, cluster
                            ),
                        0, # from_tree(
                        #     f'SLR{slr}_cluster_spike', event, card, cluster
                        #     ),
                        from_tree(
                            f'SLR{slr}_cluster_brems', event, card, cluster
                            ),
                        0, # from_tree(
                        #     f'SLR{slr}_cluster_satur', event, card, cluster
                        #     ),
                        from_tree(
                            f'SLR{slr}_cluster_spare', event, card, cluster
                            ),
                        from_tree(
                            f'ECALUnclusteredSLR{slr}_tower_et',
                            event, card, ecal_tower
                            ),
                        from_tree(
                            f'ECALUnclusteredSLR{slr}_tower_timing',
                            event, card, ecal_tower
                            ),
                        from_tree(
                            f'ECALUnclusteredSLR{slr}_tower_spike',
                            event, card, ecal_tower
                            ),
                        hcal_et,
                        hcal_fb,
                        signal_type
                        ]
