from cluster_classification.cluster_classification_dataset import process_clusters
from cluster_classification.signal_type import SignalType
from typing import Tuple
import unittest

from parameterized import parameterized
import numpy as np
import jax.numpy as jnp

from cleverlogger import CleverLogger
from cluster_classification import cluster_classification_dataset


logger = CleverLogger(__name__)

logger.log_start_load_module()


class TestClusterClassificationDataset(unittest.TestCase):
    """Tests for the class `ClusterClassifcationDataset`

    Extends: `unittest.TestCase`

    Public methods:
    - `test_ccd__get_ecal_tower__in_bounds`
    - `test_ccd__get_hcal_location__in_bounds`
    - `test_ccd__get_ecal_tower__correctness` (parameterized)
    - `test_ccd__get_hcal_location__correctness` (parameterized)
    - `test_ccd__cluster_generator__correctness`
    """

    def test_ccd__get_ecal_tower__in_bounds(self):
        """Tests if `get_ecal_tower` produces valid indices on valid input."""
        logger.log_enter_function("get_ecal_tower_in_bounds")
        logger.log_preloop("i_phi_for_loop")
        for i_phi in range(6):
            logger.log_iteration_head(i_phi=i_phi)
            # {0,1}
            logger.log_preloop("i_eta_for_loop")
            for i_eta in range(2):
                logger.log_iteration_head(i_eta=i_eta)
                slr0 = cluster_classification_dataset.get_ecal_tower(0, i_eta, i_phi)
                slr0 = slr0.item()
                self.assertIs(type(slr0), int)
                self.assertLess(slr0, 12)  # SLR0 has only 12 towers
                self.assertGreaterEqual(slr0, 0)
                logger.log_iteration_tail()
            logger.log_postloop("i_eta_for_loop")

            logger.log_preloop("slr_i_eta_for_loops")
            for slr in range(3):
                # {2,3,4,5,6}
                # {7,8,9,10,11}
                # {12,13,14,15,16}
                for i_eta in range(2 + 5 * slr, 7 + 5 * slr):
                    logger.log_iteration_head(slr=slr, i_eta=i_eta)
                    slri = cluster_classification_dataset.get_ecal_tower(
                        slr + 1, i_eta, i_phi
                    )
                    slri = slri.item()
                    self.assertIs(type(slri), int)
                    self.assertLess(slri, 30)
                    self.assertGreaterEqual(slri, 0)
                    logger.log_iteration_tail()
            logger.log_postloop("slr_i_eta_for_loop")

        logger.log_exit_function("get_ecal_tower_in_bounds")

    def test_ccd__get_hcal_location__in_bounds(self):
        """Tests if `get_hcal_location` produces valid indices on valid input."""
        logger.log_enter_function("get_ecal_tower_in_bounds")
        logger.log_preloop("i_phi_card_for_loops")
        for i_phi in range(6):
            for card in range(24):
                logger.log_iteration_head(i_phi=i_phi, card=card)
                # All i_eta > 17 should return None
                info = cluster_classification_dataset.get_hcal_location(card, 17, i_phi)
                self.assertEqual(info, (-1, -1))

                # If i_eta < 8, then data should be in links 5 or 7
                logger.log_preloop("small_i_eta_for_loop")
                for i_eta in range(8):
                    logger.log_iteration_head(i_eta=i_eta)
                    info = cluster_classification_dataset.get_hcal_location(
                        card, i_eta, i_phi
                    )
                    self.assertIsNotNone(info)
                    tower, link = info
                    tower = tower.item()
                    link = link.item()
                    self.assertIs(type(tower), int)
                    self.assertIs(type(link), int)
                    self.assertLess(tower, 32)
                    self.assertGreaterEqual(tower, 0)
                    self.assertIn(link, {5, 7})
                    logger.log_iteration_tail()
                logger.log_postloop("small_i_eta_for_loop")

                # if i_eta > 7, data should be in links 6 or 8
                logger.log_preloop("large_i_eta_for_loop")
                for i_eta in range(8, 16):
                    logger.log_iteration_head(i_eta=i_eta)
                    info = cluster_classification_dataset.get_hcal_location(
                        card, i_eta, i_phi
                    )
                    self.assertIsNotNone(info)
                    tower, link = info
                    tower = tower.item()
                    link = link.item()
                    self.assertIs(type(tower), int)
                    self.assertIs(type(link), int)
                    self.assertLess(tower, 32)
                    self.assertGreaterEqual(tower, 0)
                    self.assertIn(link, {6, 8})
                    logger.log_iteration_tail()
                logger.log_postloop("large_i_eta_for_loop")
                logger.log_iteration_tail()
        logger.log_postloop("i_phi_card_for_loops")

    @parameterized.expand(
        [
            # TODO make the test coverage as good as hcal_location
            ["SLR0", 0, 1, 0, 6],
            ["SLR1a", 1, 3, 1, 7],
            ["SLR1b", 1, 6, 2, 26],
            ["SLR2a", 2, 10, 3, 21],
            ["SLR2b", 2, 9, 4, 16],
            ["SLR3a", 3, 13, 5, 11],
            ["SLR3b", 3, 16, 0, 24],
        ]
    )
    def test_ccd__get_ecal_tower__correctness(
        self, name: str, slr: int, i_eta: int, i_phi: int, tower: int
    ):
        """Tests `get_ecal_tower` on select slr, i_eta, and i_phi indices.

        Parameterized Test:
        - `slr`: the SLR index to pass to `get_ecal_tower`
        - `i_eta`: the i eta position to pass to `get_ecal_tower`
        - `i_phi`: the i phi position to pass to `get_ecal_tower`

        Ensures `get_ecal_tower` produces correct values for select indices.
        As the function is algebraic, it is assumed that all inputs should work.

        Does **not** test what happens when given bad input.
        """
        logger.log_enter_function(
            "get_ecal_tower_correctness",
            name=name,
            slr=slr,
            i_eta=i_eta,
            i_phi=i_phi,
            tower=tower,
        )
        self.assertEqual(
            tower, cluster_classification_dataset.get_ecal_tower(slr, i_eta, i_phi)
        )
        logger.log_exit_function("get_ecal_tower_correctness")

    @parameterized.expand(
        [
            # Covers all i_eta, all i_phi, and all control flow points
            # Nicely done, two tests per condition!
            ["high link/low phi/low eta 1", 0, 0, 0, (2 + 4 * 0, 7)],
            ["high link/low phi/low eta 2", 3, 1, 1, (3 + 4 * 1, 7)],
            ["high link/low phi/high eta 1", 4, 8, 0, (2 + 4 * 0, 8)],
            ["high link/low phi/high eta 2", 7, 9, 1, (3 + 4 * 1, 8)],
            ["high link/high phi/low eta 1", 8, 2, 2, (0 + 4 * 2, 5)],
            ["high link/high phi/low eta 2", 11, 3, 3, (1 + 4 * 3, 5)],
            ["high link/high phi/high eta 1", 12, 10, 4, (2 + 4 * 2, 6)],
            ["high link/high phi/high eta 2", 15, 11, 5, (3 + 4 * 3, 6)],
            ["low link/low phi/low eta 1", 1, 4, 0, (0 + 4 * 4, 5)],
            ["low link/low phi/low eta 2", 2, 5, 1, (1 + 4 * 5, 5)],
            ["low link/low phi/high eta 1", 5, 12, 2, (2 + 4 * 4, 6)],
            ["low link/low phi/high eta 2", 6, 13, 3, (3 + 4 * 5, 6)],
            ["low link/high phi/low eta 1", 9, 6, 4, (0 + 4 * 6, 7)],
            ["low link/high phi/low eta 2", 10, 7, 5, (1 + 4 * 7, 7)],
            ["low link/high phi/high eta 1", 13, 14, 4, (0 + 4 * 6, 8)],
            ["low link/high phi/high eta 2", 14, 15, 5, (1 + 4 * 7, 8)],
        ]
    )
    def test_ccd__get_hcal_location__correctness(
        self, name: str, card: int, i_eta: int, i_phi: int, location: Tuple[int, int]
    ):
        """Tests `get_hcal_location` on select indices

        Parameterized Test:
        - `card`: The RCT card to pass to `get_hcal_location`
        - `i_eta`: the i eta position to pass to `get_hcal_location`
        - `i_phi`: the i phi position to pass to `get_hcal_location`

        Ensures `get_hcal_location produces` correct values for select indices.
        As the function is algebraic, it is assumed that all inputs should work.

        Does **not** test what happens when given bad input.
        """
        logger.log_enter_function(
            "get_hcal_location_correctness",
            name=name,
            card=card,
            i_eta=i_eta,
            i_phi=i_phi,
            location=location,
        )
        self.assertEqual(
            location,
            cluster_classification_dataset.get_hcal_location(card, i_eta, i_phi),
        )
        logger.log_exit_function("get_hcal_location_correctness")

    def test_ccd__process_cluster__correctness(self):
        slrs = [0, 1, 2, 3, 2, 3]
        cards = [0, 3, 6, 9, 12, 15]
        cluster_etas = [0, 12, 54, 76, 37, 84]
        cluster_phis = [10, 1, 16, 23, 6, 29]
        ecal_towers = [2, 0, 21, 22, 1, 29]
        hcal_links = [5, 7, 6, 8, 7]
        hcal_towers = [0, 10, 11, 28, 31]

        unclustered_ets = np.zeros((6, 30), dtype=int)
        unclustered_timings = np.zeros((6, 30), dtype=int)
        unclustered_spikes = np.zeros((6, 30), dtype=int)

        for i in range(6):
            unclustered_ets[i][ecal_towers[i]] = 1
            unclustered_timings[i][ecal_towers[i]] = 1
            unclustered_spikes[i][ecal_towers[i]] = 1

        hcal_ets = np.zeros((4, 6, 32), dtype=int)
        hcal_fbs = np.zeros((4, 6, 32), dtype=int)

        for i in range(5):
            hcal_ets[hcal_links[i] - 5][i][hcal_towers[i]] = 1
            hcal_fbs[hcal_links[i] - 5][i][hcal_towers[i]] = 1
        clusters = process_clusters(
            jnp.array(slrs),
            jnp.array([1 for _ in range(6)]),
            jnp.array(cards),
            jnp.array(cluster_etas),
            jnp.array(cluster_phis),
            jnp.array([1 for _ in range(6)]),
            jnp.array([1 for _ in range(6)]),
            jnp.array([1 for _ in range(6)]),
            jnp.array([1 for _ in range(6)]),
            jnp.array([1 for _ in range(6)]),
            jnp.array([1 for _ in range(6)]),
            jnp.array([1 for _ in range(6)]),
            jnp.array([1 for _ in range(6)]),
            jnp.array([1 for _ in range(6)]),
            jnp.array(unclustered_ets),
            jnp.array(unclustered_timings),
            jnp.array(unclustered_spikes),
            jnp.array(hcal_ets[5 - 5]),
            jnp.array(hcal_fbs[5 - 5]),
            jnp.array(hcal_ets[6 - 5]),
            jnp.array(hcal_fbs[6 - 5]),
            jnp.array(hcal_ets[7 - 5]),
            jnp.array(hcal_fbs[7 - 5]),
            jnp.array(hcal_ets[8 - 5]),
            jnp.array(hcal_fbs[8 - 5]),
        )
        clusters = np.array(clusters)
        comparisons = np.ones((6, 19), dtype=int)
        for i in range(6):
            comparisons[i][0] = slrs[i]
            comparisons[i][2] = cards[i]
            comparisons[i][3] = cluster_etas[i]
            comparisons[i][4] = cluster_phis[i]
        comparisons[-1][-1] = -1
        comparisons[-1][-2] = -1
        np.testing.assert_array_equal(clusters, comparisons)

    def test_ccd__get_labels(self):
        """Test that the label reducer works as anticipated."""
        signal_to_label, labels = cluster_classification_dataset.get_labels(
            {
                ("", SignalType.BACKGROUND),
                ("", SignalType.HADRONIC),
                ("", SignalType.BACKGROUND),
            }
        )

        self.assertIn(SignalType.BACKGROUND, signal_to_label.keys())
        self.assertIn(SignalType.HADRONIC, signal_to_label.keys())
        self.assertEqual(
            labels[signal_to_label[SignalType.BACKGROUND]], (SignalType.BACKGROUND, 0)
        )
        self.assertEqual(
            labels[signal_to_label[SignalType.HADRONIC]], (SignalType.HADRONIC, 0)
        )


logger.log_end_load_module()
