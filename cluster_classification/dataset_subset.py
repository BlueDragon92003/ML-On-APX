from __future__ import annotations
from typing import Tuple, Set, Self, Union

from cleverlogger import CleverLogger
from cluster_classification.signal_type import SignalType

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
        logger.log_enter_function(
            "ds_init",
            bitstring=bitstring,
            filename=filename,
            data_type=data_type,
            data=data,
        )
        self.bitstring = bitstring
        logger.log_open_control_flow("creation_type_if_statement")
        if data is None:
            logger.log_control_element("ThenBranch")
            if filename is None:
                raise ValueError("Must provide a filename")
            if data_type is None:
                raise ValueError("Must provide a filename")
            self.data = {(filename, data_type)}
        else:
            logger.log_control_element("ElseBranch")
            self.data = data
        logger.log_exit_function("ds_init")

    def __or__(self, other: Self) -> DatasetSubset:
        """Combine with another available dataset into a new, larger dataset."""
        logger.log_enter_function("ds_or", other=other)
        bitstring = self.bitstring | other.bitstring
        data = self.data | other.data
        retval = DatasetSubset(bitstring, data=data)
        logger.log_function_exit_type("return", retval=retval)
        logger.log_exit_function("ds_or")
        return retval

    def get_hex(self) -> str:
        """Returns a hex ID for this dataset. Enables pickle identification."""
        logger.log_enter_function("get_hex")
        bitstring = self.bitstring
        string = ""
        logger.log_open_control_flow("hex_for_loop")
        for _ in range(8):
            logger.log_open_control_flow("Iteration")
            string = hex(bitstring & 0b1111)[2:] + string
            bitstring = bitstring >> 4
            logger.log_close_control_flow("Iteration")
        logger.log_close_control_flow("hex_for_loop")
        logger.log_function_exit_type("return", retval=string)
        logger.log_exit_function("get_hex")
        return string

    def get_data(self) -> Set[Tuple[str, SignalType]]:
        """Return all filenames and corresponding signal types in this DsSs."""
        logger.log_micro_function("get_data", "return")
        return self.data

    def __len__(self) -> int:
        """Returns the number of datasets in this DsSs"""
        out = len(self.data)
        logger.log_micro_function("ds_len", "return", retval=out)
        return out

    DOUBLE_ELECTRON: DatasetSubset


DatasetSubset.DOUBLE_ELECTRON = DatasetSubset(
    0b0000_0000_0000_0000_0000_0000_0000_0001,
    "double_electron.root",
    SignalType.BACKGROUND,
)

logger.log_end_load_module()
