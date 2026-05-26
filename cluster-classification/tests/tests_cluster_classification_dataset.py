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

    def data_for__test_data_wrapper__cluster_generator_correctness(feature, event, card, final):
        """
        A simple method that serves as a datasource for the cluster_generator.
        It allows for control over the data provided to the test and improves
        test duration.
        """

        # 6 total clusters will be generaed by this process:
        #   - 4 clusters are in event 0, 2 in event 1
        #   - Cards 0, 3, 6, 9, 12, 15 are tested
        #   - All SLR numbers are tested
        #   - All link numbers are tested
        #   - Cluster positions 0, 2, 4, 5, 6, 7 are tested
        #   - All i_phi tower values are tested
        #   - i_eta tower values 0, 2, 7, 10, 15, 16 are tested
        #
        #   - ecal towers 0, 3, 4, 19, 21, 29
        #   - hcal towers 0, 10, 11, 29, 30, and 'none'

        """ TEST CLUSTER 0 -- SLR0, Link 5, high link
             0: cluster_et              1
             1: cluster_seed            2
             2: cluster_et5x5           3
             3: cluster_et2x5           4
             4: cluster_timing          5
             5: cluster_spike           6
             6: cluster_brems           7
             7: cluster_satur           8
             8: cluster_spare           9
             9: ecal_tower_et           10
            10: ecal_tower_timing       11
            11: ecal_tower_spike        12
            12: hcal_et                 13
            13: hcal_fb                 14
            
            Event: 0
            Card: 0
            SLR: 0
            Cluster position: 0
            Link: 5
            hcal_tower: 0
            ecal_tower: 3
            i_eta: 0 (tower)
            i_phi: 2 (tower)
            """  
        if event == 0 and card == 0:
            match feature:
                case 'SLR0_cluster_eta':
                    if final == 0: # cluster position
                        return 0
                case 'SLR0_cluster_phi':
                    if final == 0:
                        return 10
                
                case 'SLR0_cluster_energy':
                    if final == 0: # cluster position
                        return 1
                case 'SLR0_cluster_seed_energy':
                    if final == 0:
                        return 2
                case 'SLR0_cluster_et5x5':
                    if final == 0:
                        return 3
                case 'SLR0_cluster_et2x5':
                    if final == 0:
                        return 4
                case 'SLR0_cluster_timing':
                    if final == 0:
                        return 5
                case 'SLR0_cluster_spike':
                    if final == 0:
                        return 6
                case 'SLR0_cluster_brems':
                    if final == 0:
                        return 7
                case 'SLR0_cluster_satur':
                    if final == 0:
                        return 8
                case 'SLR0_cluster_spare':
                    if final == 0:
                        return 9
                
                case 'ECALUnclusteredSLR0_tower_et':
                    if final == 3: # ecal tower
                        return 10
                case 'ECALUnclusteredSLR0_tower_timing':
                    if final == 3:
                        return 11
                case 'ECALUnclusteredSLR0_tower_spike':
                    if final == 3:
                        return 12

                case 'HCAL5_tower_et':
                    if final == 0: # hcal tower
                        return 13
                case 'HCAL5_tower_fb':
                    if final == 0:
                        return 14

                case _:
                    return 0
            return 0

        """ TEST CLUSTER 1 -- SLR1, Link 7, high link
             0: cluster_et              2
             1: cluster_seed            3
             2: cluster_et5x5           5
             3: cluster_et2x5           7
             4: cluster_timing          11
             5: cluster_spike           13
             6: cluster_brems           17
             7: cluster_satur           23
             8: cluster_spare           29
             9: ecal_tower_et           31
            10: ecal_tower_timing       37
            11: ecal_tower_spike        41
            12: hcal_et                 43
            13: hcal_fb                 47
            
            Event: 0
            Card: 3
            SLR: 1
            Cluster position: 2
            Link: 7
            hcal_tower: 10
            ecal_tower: 0
            i_eta: 2 (tower)
            i_phi: 0 (tower)
            """
        if event == 0 and card == 3:
            match feature:
                case 'SLR1_cluster_eta':
                    if final == 2: # cluster position
                        return 12
                case 'SLR1_cluster_phi':
                    if final == 2:
                        return 1
                
                case 'SLR1_cluster_energy':
                    if final == 2: # cluster position
                        return 2
                case 'SLR1_cluster_seed_energy':
                    if final == 2:
                        return 3
                case 'SLR1_cluster_et5x5':
                    if final == 2:
                        return 5
                case 'SLR1_cluster_et2x5':
                    if final == 2:
                        return 7
                case 'SLR1_cluster_timing':
                    if final == 2:
                        return 11
                case 'SLR1_cluster_spike':
                    if final == 2:
                        return 13
                case 'SLR1_cluster_brems':
                    if final == 2:
                        return 17
                case 'SLR1_cluster_satur':
                    if final == 2:
                        return 23
                case 'SLR1_cluster_spare':
                    if final == 2:
                        return 29
                
                case 'ECALUnclusteredSLR1_tower_et':
                    if final == 0: # ecal tower
                        return 31
                case 'ECALUnclusteredSLR1_tower_timing':
                    if final == 0:
                        return 37
                case 'ECALUnclusteredSLR1_tower_spike':
                    if final == 0:
                        return 41

                case 'HCAL7_tower_et':
                    if final == 10: # hcal tower
                        return 43
                case 'HCAL7_tower_fb':
                    if final == 10:
                        return 47

                case _:
                    return 0
            return 0

        """ TEST CLUSTER 2 -- SLR2 , Link6, low link
             0: cluster_et              1
             1: cluster_seed            1
             2: cluster_et5x5           2
             3: cluster_et2x5           3
             4: cluster_timing          5
             5: cluster_spike           8
             6: cluster_brems           13
             7: cluster_satur           21
             8: cluster_spare           34
             9: ecal_tower_et           55
            10: ecal_tower_timing       -39
            11: ecal_tower_spike        16
            12: hcal_et                 -23
            13: hcal_fb                 -7
            
            Event: 0
            Card: 6
            SLR: 2
            Cluster position: 4
            Link: 6
            hcal_tower: 11
            ecal_tower: 21
            i_eta: 10 (tower)
            i_phi: 3 (tower)
            """
        if event == 0 and card == 6:
            match feature:
                case 'SLR2_cluster_eta':
                    if final == 4: # cluster position
                        return 54
                case 'SLR2_cluster_phi':
                    if final == 4:
                        return 16
                
                case 'SLR2_cluster_energy':
                    if final == 4: # cluster position
                        return 1
                case 'SLR2_cluster_seed_energy':
                    if final == 4:
                        return 1
                case 'SLR2_cluster_et5x5':
                    if final == 4:
                        return 2
                case 'SLR2_cluster_et2x5':
                    if final == 4:
                        return 3
                case 'SLR2_cluster_timing':
                    if final == 4:
                        return 5
                case 'SLR2_cluster_spike':
                    if final == 4:
                        return 8
                case 'SLR2_cluster_brems':
                    if final == 4:
                        return 13
                case 'SLR2_cluster_satur':
                    if final == 4:
                        return 21
                case 'SLR2_cluster_spare':
                    if final == 4:
                        return 34
                
                case 'ECALUnclusteredSLR2_tower_et':
                    if final == 21: # ecal tower
                        return 55
                case 'ECALUnclusteredSLR2_tower_timing':
                    if final == 21:
                        return -39
                case 'ECALUnclusteredSLR2_tower_spike':
                    if final == 21:
                        return 16

                case 'HCAL7_tower_et':
                    if final == 11: # hcal tower
                        return -23
                case 'HCAL7_tower_fb':
                    if final == 11:
                        return -7

                case _:
                    return 0
            return 0

        """ TEST CLUSTER 3 -- SLR3, Link8, high link
             0: cluster_et              1
             1: cluster_seed            8
             2: cluster_et5x5           2
             3: cluster_et2x5           9
             4: cluster_timing          3
             5: cluster_spike           10
             6: cluster_brems           4
             7: cluster_satur           11
             8: cluster_spare           5
             9: ecal_tower_et           12
            10: ecal_tower_timing       6
            11: ecal_tower_spike        13
            12: hcal_et                 7
            13: hcal_fb                 14
            
            Event: 0
            Card: 9
            SLR: 3
            Cluster position: 5
            Link: 8
            hcal_tower: 29
            ecal_tower: 19
            i_eta: 15 (tower)
            i_phi: 1 (tower)
            """
        if event == 0 and card == 9:
            match feature:
                case 'SLR3_cluster_eta':
                    if final == 5: # cluster position
                        return 76 # trombones
                case 'SLR3_cluster_phi':
                    if final == 5:
                        return 8
                
                case 'SLR3_cluster_energy':
                    if final == 5: # cluster position
                        return 1
                case 'SLR3_cluster_seed_energy':
                    if final == 5:
                        return 8
                case 'SLR3_cluster_et5x5':
                    if final == 5:
                        return 2
                case 'SLR3_cluster_et2x5':
                    if final == 5:
                        return 9
                case 'SLR3_cluster_timing':
                    if final == 5:
                        return 3
                case 'SLR3_cluster_spike':
                    if final == 5:
                        return 10
                case 'SLR3_cluster_brems':
                    if final == 5:
                        return 4
                case 'SLR3_cluster_satur':
                    if final == 5:
                        return 11
                case 'SLR3_cluster_spare':
                    if final == 5:
                        return 5
                
                case 'ECALUnclusteredSLR3_tower_et':
                    if final == 19: # ecal tower
                        return 12
                case 'ECALUnclusteredSLR3_tower_timing':
                    if final == 19:
                        return 6
                case 'ECALUnclusteredSLR3_tower_spike':
                    if final == 19:
                        return 13

                case 'HCAL8_tower_et':
                    if final == 29: # hcal tower
                        return 7
                case 'HCAL8_tower_fb':
                    if final == 29:
                        return 14

                case _:
                    return 0
            return 0
        
        """ TEST CLUSTER 4 -- SLR2, Link 5, high link
             0: cluster_et              14
             1: cluster_seed            13
             2: cluster_et5x5           12
             3: cluster_et2x5           11
             4: cluster_timing          10
             5: cluster_spike           9
             6: cluster_brems           8
             7: cluster_satur           7
             8: cluster_spare           6
             9: ecal_tower_et           5
            10: ecal_tower_timing       4
            11: ecal_tower_spike        3
            12: hcal_et                 2
            13: hcal_fb                 1
            
            Event: 1
            Card: 12
            SLR: 2
            Cluster position: 6
            Link: 5
            hcal_tower: 30
            ecal_tower: 4
            i_eta: 7 (tower)
            i_phi: 4 (tower)
            """
        if event == 1 and card == 12:
            match feature:
                case 'SLR2_cluster_eta':
                    if final == 6: # cluster position
                        return 37
                case 'SLR2_cluster_phi':
                    if final == 6:
                        return 23
                
                case 'SLR2_cluster_energy':
                    if final == 6: # cluster position
                        return 14
                case 'SLR2_cluster_seed_energy':
                    if final == 6:
                        return 13
                case 'SLR2_cluster_et5x5':
                    if final == 6:
                        return 12
                case 'SLR2_cluster_et2x5':
                    if final == 6:
                        return 11
                case 'SLR2_cluster_timing':
                    if final == 6:
                        return 10
                case 'SLR2_cluster_spike':
                    if final == 6:
                        return 9
                case 'SLR2_cluster_brems':
                    if final == 6:
                        return 8
                case 'SLR2_cluster_satur':
                    if final == 6:
                        return 7
                case 'SLR2_cluster_spare':
                    if final == 6:
                        return 6
                
                case 'ECALUnclusteredSLR2_tower_et':
                    if final == 4: # ecal tower
                        return 5
                case 'ECALUnclusteredSLR2_tower_timing':
                    if final == 4:
                        return 4
                case 'ECALUnclusteredSLR2_tower_spike':
                    if final == 4:
                        return 3

                case 'HCAL5_tower_et':
                    if final == 30: # hcal tower
                        return 2
                case 'HCAL5_tower_fb':
                    if final == 30:
                        return 1

                case _:
                    return 0
            return 0

        """ TEST CLUSTER 5 -- SLR3, no link
             0: cluster_et              11
             1: cluster_seed            9
             2: cluster_et5x5           7
             3: cluster_et2x5           5
             4: cluster_timing          3
             5: cluster_spike           1
             6: cluster_brems           0
             7: cluster_satur           2
             8: cluster_spare           4
             9: ecal_tower_et           6
            10: ecal_tower_timing       8
            11: ecal_tower_spike        10
            12: hcal_et                 -1
            13: hcal_fb                 -1
            
            Event: 1
            Card: 15
            SLR: 3
            Cluster position: 7
            Link: N/A
            hcal_tower: N/A
            ecal_tower: 29
            i_eta: 16 (tower)
            i_phi: 5 (tower)
            """
        if event == 1 and card == 15:
            match feature:
                case 'SLR3_cluster_eta':
                    if final == 7: # cluster position
                        return 84
                case 'SLR3_cluster_phi':
                    if final == 7:
                        return 29
                
                case 'SLR3_cluster_energy':
                    if final == 7: # cluster position
                        return 11
                case 'SLR3_cluster_seed_energy':
                    if final == 7:
                        return 9
                case 'SLR3_cluster_et5x5':
                    if final == 7:
                        return 7
                case 'SLR3_cluster_et2x5':
                    if final == 7:
                        return 5
                case 'SLR3_cluster_timing':
                    if final == 7:
                        return 3
                case 'SLR3_cluster_spike':
                    if final == 7:
                        return 1
                case 'SLR3_cluster_brems':
                    if final == 7:
                        return 0
                case 'SLR3_cluster_satur':
                    if final == 7:
                        return 2
                case 'SLR3_cluster_spare':
                    if final == 7:
                        return 4
                
                case 'ECALUnclusteredSLR3_tower_et':
                    if final == 29: # ecal tower
                        return 6
                case 'ECALUnclusteredSLR3_tower_timing':
                    if final == 29:
                        return 8
                case 'ECALUnclusteredSLR3_tower_spike':
                    if final == 29:
                        return 10

                case _:
                    return 0
            return 0

        return 0

    def test_data_wrapper__cluster_generator_correctness(self):
        

        pass
