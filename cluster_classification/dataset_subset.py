from __future__ import annotations
from typing import Tuple, Set, Self, Union

from cluster_classification.signal_type import SignalType
from cluster_classification.classification_logger import CleverLogger

logger = CleverLogger(__name__)

logger.log_start_load_module()

class DatasetSubset:
    data: Set[Tuple[str, SignalType]]
    """Represents a subset of data sources data is pulled from.

    Public methods:
    - `get_hex`
    - `get_data`
    
    Behaviors:
    - `len(self)`: Extracts the number of distinct source files in this
                    `DatasetSubset`
    - `self | other`: Composes two `DatasetSubsets` into a larger one.
    """

    def __init__(
        self,
        bitstring: int,
        filename: Union[str, None] = None,
        data_type: Union[SignalType, None] = None,
        data: Union[Set[Tuple[str, SignalType]], None] = None,
    ):
        """Create a new subset from a ROOT file and a signal type or data.

        Subsets should either be created as constants later in this file using
        filenames, unique bitstrings, and signal types; or by combining other
        `DatasetSubsets` using the or (`|`) operator.

        Arguments:
        - `bitstring`: The bitstring this DsSs should use
        - `filename`: The file name for a root file
        - `data_type`: The signal type the root file represents
        - `data`: A set of tuples of filenames and signal types this DsSs should
                    contain.
        ONLY `data` OR (`filename` AND `data_type`) should be used.
        """
        logger.log_trace(
            f"<dataset_subset.DatasetSubset.__init__ bitstring={bitstring} filename={filename} data_type={data_type} data={data}>"
        )
        self.bitstring = bitstring
        if data is None:
            if filename is None:
                raise ValueError("Must provide a filename")
            if data_type is None:
                raise ValueError("Must provide a filename")
            self.data = {(filename, data_type)}
        else:
            self.data = data
        logger.log_trace("</dataset_subset.DatasetSubset.__init__>")

    def __or__(self, other: Self) -> DatasetSubset:
        """Combine with another available dataset into a new, larger dataset."""
        logger.log_trace("<dateset_subset.DatasetSubset.__or__ />")
        bitstring = self.bitstring | other.bitstring
        data = self.data | other.data
        return DatasetSubset(bitstring, data=data)

    def get_hex(self) -> str:
        """Returns a hex ID for this dataset. Enables pickle identification."""
        bitstring = self.bitstring
        string = ""
        for _ in range(8):
            string = hex(bitstring & 0b1111)[2:] + string
            bitstring = bitstring >> 4
        logger.log_trace(f"<dateset_subset.DatasetSubset.get_hex ret={string} />")
        return string

    def get_data(self) -> Set[Tuple[str, SignalType]]:
        """Return all filenames and corresponding signal types in this DsSs."""
        logger.log_trace("<dateset_subset.DatasetSubset.get_data />")
        return self.data

    def __len__(self) -> int:
        """Returns the number of datasets in this DsSs"""
        logger.log_trace("<dateset_subset.DatasetSubset.__len__ />")
        return len(self.data)

    DOUBLE_ELECTRON: DatasetSubset


DOUBLE_ELECTRON = DatasetSubset(
    0b0000_0000_0000_0000_0000_0000_0000_0001,
    "double_electron.root",
    SignalType.BACKGROUND,
)

logger.log_end_load_module()
