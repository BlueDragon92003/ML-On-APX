import cluster_classification_dataset

import unittest

from parameterized import parameterized

class TestClusterClassificationDataset(unittest.TestCase):

    def test_data_wrapper__get_ecal_tower_in_bounds(self):
        """
        Ensure that the ECAL tower function produces only in-range indices on
        valid input.
        """
        for i_phi in range(6):
            # {0,1}
            for i_eta in range(2):
                slr0 = cluster_classification_dataset.get_ecal_tower(0, i_eta, i_phi)
                self.assertIs( type(slr0), int )
                self.assertLess( slr0, 12 ) # SLR0 has only 12 towers
                self.assertGreaterEqual( slr0, 0 )

            for slr in range(3):
                # {2,3,4,5,6}
                # {7,8,9,10,11}
                # {12,13,14,15,16}
                for i_eta in range(2+5*slr,7+5*slr):
                    slri = cluster_classification_dataset.get_ecal_tower(slr+1, i_eta, i_phi)
                    self.assertIs( type(slri), int )
                    self.assertLess( slri, 30 )
                    self.assertGreaterEqual( slri, 0 )

    def test_data_wrapper__get_hcal_location_in_bounds(self):
        """
        Ensure that the HCAL location function produces only in-range indices.
        """
        for i_phi in range(6):
            for card in range(24):
                # All i_eta > 17 should return None
                info = cluster_classification_dataset.get_hcal_location(card, 17, i_phi)
                self.assertIsNone(info)
                
                # If i_eta < 8, then data should be in links 5 or 7
                for i_eta in range(8):
                    info = cluster_classification_dataset.get_hcal_location(card, i_eta, i_phi)
                    self.assertIsNotNone(info)
                    tower, link = info
                    self.assertIs( type(tower), int )
                    self.assertIs( type(link), int )
                    self.assertLess( tower, 32 )
                    self.assertGreaterEqual( tower, 0 )
                    self.assertIn( link, {5,7} )
                
                # if i_eta > 7, data should be in links 6 or 8
                for i_eta in range(8,16):
                    info = cluster_classification_dataset.get_hcal_location(card, i_eta, i_phi)
                    self.assertIsNotNone(info)
                    tower, link = info
                    self.assertIs( type(tower), int )
                    self.assertIs( type(link), int )
                    self.assertLess( tower, 32 )
                    self.assertGreaterEqual( tower, 0 )
                    self.assertIn( link, {6,8} )

    # TODO create correctness tests

    @parameterized.expand([
        # TODO make the test coverage as good as hcal_location
        [0,1,0,6],
        [1,3,1,7],
        [1,6,2,26],
        [2,10,3,21],
        [2,9,4,16],
        [3,13,5,11],
        [3,16,0,24],
    ])
    def test_data_wrapper__get_ecal_tower_correctness(self, slr, i_eta, i_phi, tower):
        """
        Ensures get_ecal_tower produces correct values for some indices.
        Implies correctness for the rest of them.

        Does not test what happens when given bad input.
        """
        self.assertEqual(
            tower,
            cluster_classification_dataset.get_ecal_tower(slr, i_eta, i_phi)
        )

    @parameterized.expand([
        # Covers all i_eta, all i_phi, and all control flow points
        # Nicely done, two tests per condition!
        [0,0,0,(2+4*0,7)], # high link/low phi/low eta 1
        [3,1,1,(3+4*1,7)], # high link/low phi/low eta 2
        [4,8,0,(2+4*0,8)], # high link/low phi/high eta 1
        [7,9,1,(3+4*1,8)], # high link/low phi/high eta 2
        [8,2,2,(0+4*2,5)], # high link/high phi/low eta 1
        [11,3,3,(1+4*3,5)], # high link/high phi/low eta 2
        [12,10,4,(2+4*2,6)], # high link/high phi/high eta 1
        [15,11,5,(3+4*3,6)], # high link/high phi/high eta 2
        [1,4,0,(0+4*4,5)], # low link/low phi/low eta 1
        [2,5,1,(1+4*5,5)], # low link/low phi/low eta 2
        [5,12,2,(2+4*4,6)], # low link/low phi/high eta 1
        [6,13,3,(3+4*5,6)], # low link/low phi/high eta 2
        [9,6,4,(0+4*6,7)], # low link/high phi/low eta 1
        [10,7,5,(1+4*7,7)], # low link/high phi/low eta 2
        [13,14,4,(0+4*6,8)], # low link/high phi/high eta 1
        [14,15,5,(1+4*7,8)], # low link/high phi/high eta 2
    ])
    def test_data_wrapper__get_hcal_tower_correctness(self, card, i_eta, i_phi, location):
        """
        Ensures get_hcal_location produces correct values for some indices.
        Implies correctness for the rest of them.

        Does not test what happens when given bad input.
        """
        self.assertEqual(
            location,
            cluster_classification_dataset.get_hcal_location(card, i_eta, i_phi)
        )
