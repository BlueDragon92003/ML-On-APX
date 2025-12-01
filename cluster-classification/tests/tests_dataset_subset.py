import unittest
from parameterized import parameterized

from dataset_subset import DatasetSubset
from cluster import ClusterType

class TestDatasetSubset(unittest.TestCase):

    def test_dataset_subset__instantiation_data(self):
        """
        Test that nothing blows up when a new subset is created using data.
        """
        DatasetSubset(0x0, data={("test", ClusterType.HADRONIC)})

    def test_dataset_subset__instantiation_file_type(self):
        """
        Test that nothing blows up when a new subset is created using the
        file/type method
        """
        DatasetSubset(0x0, filename="test", cluster_type=ClusterType.HADRONIC)

    def test_dataset_subset__instantiation_breaks(self):
        """Ensure that an error is raised if a subset is created improperly."""
        with self.assertRaises(ValueError):
            DatasetSubset(0x0)
        with self.assertRaises(ValueError):
            DatasetSubset(0x0, filename="test")
        with self.assertRaises(ValueError):
            DatasetSubset(0x0, cluster_type=ClusterType.ELECTROMAGNETIC)
    
    @parameterized.expand([
        ["Zero", 0, "00000000"],
        ["Max", 0xffff_ffff, "ffffffff"],
        ["Arbitrary", 0xDEAD, "0000dead"],
    ])
    def test_dataset_subset__get_hex(self, name, bitstring, expected):
        """Test that the `get_hex` method works as expected."""
        sub = DatasetSubset(bitstring, filename="test", cluster_type=ClusterType.HADRONIC)
        self.assertEqual(expected, sub.get_hex())
    
    @parameterized.expand([
        ["Empty", set()],
        ["One", {("test", ClusterType.ELECTROMAGNETIC)}],
        ["Many", {  ("zero", ClusterType.HADRONIC),
                    ("one", ClusterType.ELECTROMAGNETIC),
                    ("two", ClusterType.HADRONIC),
                    ("three", ClusterType.ELECTROMAGNETIC),
                    ("four", ClusterType.HADRONIC),
                    ("five", ClusterType.HADRONIC)
                }],
    ])
    def test_dataset_subset__get_data(self, name, data):
        """Test that the `get_data` method works as expected."""
        sub = DatasetSubset(0, data=data)
        self.assertEqual(data, sub.get_data())
    
    @parameterized.expand([
        ["Empty", set(), 0],
        ["One", {("test", ClusterType.ELECTROMAGNETIC)}, 1],
        ["Many", {  ("zero", ClusterType.HADRONIC),
                    ("one", ClusterType.ELECTROMAGNETIC),
                    ("two", ClusterType.HADRONIC),
                    ("three", ClusterType.ELECTROMAGNETIC),
                    ("four", ClusterType.HADRONIC),
                    ("five", ClusterType.HADRONIC)
                }, 6],
    ])
    def test_dataset_subset__len(self, name, data, length):
        """Test that the `len` method works as expected."""
        sub = DatasetSubset(0, data=data)
        self.assertEqual(length, len(sub))
    
    @parameterized.expand([
        ["Empty", 0, 0, "00000000"],
        ["Zero", 0xDEAD, 0, "0000dead"],
        ["Same", 0xDEAD, 0xDEAD, "0000dead"],
        ["Arbitrary", 0xABCD, 0x1234, "0000bbfd"],
    ])
    def test_dataset_subset__combo__get_hex(self, name, hex1, hex2, expected):
        """
        Test that the `get_hex` method works as expected when two sets are
        combined.
        """
        sub1 = DatasetSubset(hex1, data={})
        sub2 = DatasetSubset(hex2, data={})
        sub = sub1 | sub2
        self.assertEqual(expected, sub.get_hex())

    @parameterized.expand([
        ["Empty", set(), set()],
        ["Zero", {("test", ClusterType.ELECTROMAGNETIC)}, set()],
        ["Same", {("test", ClusterType.HADRONIC)}, {("test", ClusterType.HADRONIC)}],
        ["Different", {("test1", ClusterType.ELECTROMAGNETIC)}, {("test2", ClusterType.HADRONIC)}],
    ])
    def test_dataset_subset__combo__get_data(self, name, data1, data2):
        """
        Test that the `get_data` method works as expected when two sets are
        combined.
        """
        sub1 = DatasetSubset(0, data=data1)
        sub2 = DatasetSubset(0, data=data2)
        sub = sub1 | sub2
        expected = data1 | data2
        self.assertEqual(expected, sub.get_data())
    
    @parameterized.expand([
        ["Empty", set(), set(), 0],
        ["Zero", {("test", ClusterType.ELECTROMAGNETIC)}, set(), 1],
        ["Same", {("test", ClusterType.HADRONIC)}, {("test", ClusterType.HADRONIC)}, 1],
        ["Different", {("test1", ClusterType.ELECTROMAGNETIC)}, {("test2", ClusterType.HADRONIC)}, 2],
    ])
    def test_dataset_subset__combo__len(self, name, data1, data2, expected):
        """
        Test that the `len` method works as expected when two sets are
        combined.
        """
        sub1 = DatasetSubset(0, data=data1)
        sub2 = DatasetSubset(0, data=data2)
        sub = sub1 | sub2
        self.assertEqual(expected, len(sub))
