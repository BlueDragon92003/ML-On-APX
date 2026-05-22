import cluster_classification_dataset

import unittest

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
