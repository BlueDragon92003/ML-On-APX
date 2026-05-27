from typing import Set, Tuple
import unittest

from parameterized import parameterized

from dataset_subset import DatasetSubset
from signal_type import SignalType

# TODO typing
class TestDatasetSubset(unittest.TestCase):
    """Tests for the class `DatasetSubset`

    Extends: `unittest.TestCase`

    Public methods:
    - `test_dataset_subset__instantiation_data`
    - `test_dataset_subset__instantiation_file_type`
    - `test_dataset_subset__instantiation_breaks`
    - `test_dataset_subset__get_hex` (parameterized)
    - `test_dataset_subset__get_data` (parameterized)
    - `test_dataset_subset__len` (parameterized)
    - `test_dataset_subset__combo__get_hex` (parameterized)
    - `test_dataset_subset__combo__get_data`
    - `test_dataset_subset__combo__len`
    """

    def test_dataset_subset__instantiation_data(self):
        """Tests for errors when a subset is created using data."""
        DatasetSubset(0x0, data={("test", SignalType.HADRONIC)})

    def test_dataset_subset__instantiation_file_type(self):
        """Tests for errors when a subset is created using files and types"""
        DatasetSubset(0x0, filename="test", data_type=SignalType.HADRONIC)

    def test_dataset_subset__instantiation_breaks(self):
        """Tests if an error is raised if a subset is created improperly."""
        with self.assertRaises(ValueError):
            DatasetSubset(0x0)
        with self.assertRaises(ValueError):
            DatasetSubset(0x0, filename="test")
        with self.assertRaises(ValueError):
            DatasetSubset(0x0, data_type=SignalType.BACKGROUND)
    
    @parameterized.expand([
        ["Zero", 0, "00000000"],
        ["Max", 0xffff_ffff, "ffffffff"],
        ["Arbitrary", 0xDEAD, "0000dead"],
    ])
    def test_dataset_subset__get_hex(self,
        name: str,
        bitstring: int,
        expected: str
        ):
        """Test that the `get_hex` method works as expected.
        
        Parameterized Test:
        - `name`: Name for the parameterization
        - `bitstring`: Provided bitstring for the dataset
        - `expected`: Expected hex string for the dataset
        """
        sub = DatasetSubset(
            bitstring,
            filename="test",
            data_type=SignalType.HADRONIC
            )
        self.assertEqual(expected, sub.get_hex())
    
    @parameterized.expand([
        ["Empty", set()],
        ["One", {("test", SignalType.BACKGROUND)}],
        ["Many", {  ("zero", SignalType.HADRONIC),
                    ("one", SignalType.BACKGROUND),
                    ("two", SignalType.HADRONIC),
                    ("three", SignalType.BACKGROUND),
                    ("four", SignalType.HADRONIC),
                    ("five", SignalType.HADRONIC)
                }],
    ])
    def test_dataset_subset__get_data(self,
        name: str,
        data: Set[Tuple[str, SignalType]]
        ):
        """Test that the `get_data` method works as expected.
        
        Parameterized Test:
        - `name`: Name for the parameterization
        - `data`: Provided `DatasetSubset` data.
        """
        sub = DatasetSubset(0, data=data)
        self.assertEqual(data, sub.get_data())
    
    @parameterized.expand([
        ["Empty", set(), 0],
        ["One", {("test", SignalType.BACKGROUND)}, 1],
        ["Many", {  ("zero", SignalType.HADRONIC),
                    ("one", SignalType.BACKGROUND),
                    ("two", SignalType.HADRONIC),
                    ("three", SignalType.BACKGROUND),
                    ("four", SignalType.HADRONIC),
                    ("five", SignalType.HADRONIC)
                }, 6],
    ])
    def test_dataset_subset__len(self,
        name: str,
        data: Set[Tuple[str, SignalType]],
        length: int
        ):
        """Test that the `len` method works as expected.
        
        Parameterized Test:
        - `name`: Name for the parameterization
        - `data`: Provided `DatasetSubset` data.
        - `length`: Expected length of the data.
        """
        sub = DatasetSubset(0, data=data)
        self.assertEqual(length, len(sub))
    
    @parameterized.expand([
        ["Empty", 0, 0, "00000000"],
        ["Zero", 0xDEAD, 0, "0000dead"],
        ["Same", 0xDEAD, 0xDEAD, "0000dead"],
        ["Arbitrary", 0xABCD, 0x1234, "0000bbfd"],
    ])
    def test_dataset_subset__combo__get_hex(self,
        name: str,
        hex1: int,
        hex2: int,
        expected: str
        ):
        """Test that the `get_hex` method when two sets are combined.
        
        Parameterized Test:
        - `name`: Name for the parameterization
        - `hex1`: Provided bitstring for the first set
        - `hex2`: Provided bitstring for the second set
        - `expected`: Expected string, combining both bitstrings.
        """
        sub1 = DatasetSubset(hex1, data={})
        sub2 = DatasetSubset(hex2, data={})
        sub = sub1 | sub2
        self.assertEqual(expected, sub.get_hex())

    @parameterized.expand([
        ["Empty", set(), set()],
        ["Zero", {("test", SignalType.BACKGROUND)}, set()],
        [
            "Same",
            {("test", SignalType.HADRONIC)},
            {("test", SignalType.HADRONIC)}
            ],
        [
            "Different",
            {("test1", SignalType.BACKGROUND)},
            {("test2", SignalType.HADRONIC)}
            ],
    ])
    def test_dataset_subset__combo__get_data(self,
        name: str,
        data1: Set[Tuple[str, SignalType]],
        data2: Set[Tuple[str, SignalType]]
        ):
        """ Test that the `get_data` method when two sets are combined.
        
        Parameterized Test:
        - `name`: Name for the parameterization
        - `data1`: Provided data for the first initial set.
        - `data2`: Provided data for the second initial set.
        """
        sub1 = DatasetSubset(0, data=data1)
        sub2 = DatasetSubset(0, data=data2)
        sub = sub1 | sub2
        expected = data1 | data2
        self.assertEqual(expected, sub.get_data())
    
    @parameterized.expand([
        ["Empty", set(), set(), 0],
        ["Zero", {("test", SignalType.BACKGROUND)}, set(), 1],
        [
            "Same",
            {("test", SignalType.HADRONIC)},
            {("test", SignalType.HADRONIC)},
            1
            ],
        [
            "Different",
            {("test1", SignalType.BACKGROUND)},
            {("test2", SignalType.HADRONIC)},
            2
            ],
    ])
    def test_dataset_subset__combo__len(self,
        name: str,
        data1: Set[Tuple[str, SignalType]],
        data2: Set[Tuple[str, SignalType]],
        expected: int
        ):
        """ Test that the `len` method when two sets are combined.

        Parameterized Test:
        - `name`: Name for the parameterization
        - `data1`: Provided data for the first initial set.
        - `data2`: Provided data for the second initial set.
        - `expected`: Expected string, combining both bitstrings.
        """
        sub1 = DatasetSubset(0, data=data1)
        sub2 = DatasetSubset(0, data=data2)
        sub = sub1 | sub2
        self.assertEqual(expected, len(sub))
