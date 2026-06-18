from __future__ import annotations
from typing import Tuple, Set, Self, List

from ml_on_apx.cleverlogger import CleverLogger
from ml_on_apx.cluster_classification.signal_type import SignalType


class DatasetSubset:
    __create_key = object()

    __data: Set[Tuple[str, SignalType]]
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
        create_key: object,
        # filename: Union[str, None] = None,
        # data_type: Union[SignalType, None] = None,
        # data: Union[Set[Tuple[str, SignalType]], None] = None,
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
        self.logger = CleverLogger(__name__)
        self.logger.log_enter_function(
            "ds_init",
            bitstring=bitstring,
            # filename=filename,
            # data_type=data_type,
            # data=data,
        )
        if create_key is not DatasetSubset.__create_key:
            raise TypeError(
                "Please use the new_dataset method to create a Dataset Subset."
            )
        self.bitstring = bitstring
        # if data is None:
        #     if filename is None:
        #         raise ValueError("Must provide a filename")
        #     if data_type is None:
        #         raise ValueError("Must provide a filename")
        #     self.data = {(filename, data_type)}
        # else:
        #     self.data = data
        self.__data: Set[Tuple[str, SignalType]] = set()
        self.logger.log_exit_function("ds_init")

    @classmethod
    def new_dataset(
        cls, bitstring: int, filenames: List[str], signal_type: SignalType
    ) -> DatasetSubset:
        """Construct a new dataset from a list of filenames and a signal type"""
        logger = CleverLogger(__name__)
        logger.log_enter_function("new_dataset")
        retval = DatasetSubset(bitstring, cls.__create_key)
        data = set()
        logger.log_preloop("filenames_for_loop")
        for filename in filenames:
            logger.log_iteration_head(filename=filename)
            data.add((filename, signal_type))
            logger.log_iteration_tail()
        logger.log_postloop("filenames_for_loop")
        retval.__data = data
        logger.log_function_exit_type("return", retval=retval)
        logger.log_exit_function("new_dataset")
        return retval

    @classmethod
    def _from_raw_components(
        cls, bitstring: int, data: Set[Tuple[str, SignalType]]
    ) -> DatasetSubset:
        """Construct a new dataset from raw data"""
        logger = CleverLogger(__name__)
        logger.log_enter_function("__from_raw_components")
        retval = DatasetSubset(bitstring, cls.__create_key)
        retval.__data = data
        logger.log_function_exit_type("return", retval=retval)
        logger.log_exit_function("__from_raw_components")
        return retval

    def __or__(self, other: Self) -> DatasetSubset:
        """Combine with another available dataset into a new, larger dataset."""
        self.logger.log_enter_function("ds_or", other=other)
        bitstring = self.bitstring | other.bitstring
        data = self.__data | other.__data
        retval = DatasetSubset._from_raw_components(bitstring, data)
        self.logger.log_function_exit_type("return", retval=retval)
        self.logger.log_exit_function("ds_or")
        return retval

    def get_hex(self) -> str:
        """Returns a hex ID for this dataset. Enables pickle identification."""
        self.logger.log_enter_function("get_hex")
        bitstring = self.bitstring
        string = ""
        self.logger.log_preloop("hex_for_loop")
        for _ in range(8):
            self.logger.log_iteration_head()
            string = hex(bitstring & 0b1111)[2:] + string
            bitstring = bitstring >> 4
            self.logger.log_iteration_tail()
        self.logger.log_postloop("hex_for_loop")
        self.logger.log_function_exit_type("return", retval=string)
        self.logger.log_exit_function("get_hex")
        return string

    def get_data(self) -> Set[Tuple[str, SignalType]]:
        """Return all filenames and corresponding signal types in this DsSs."""
        self.logger.log_micro_function("get_data", "return")
        return self.__data

    def __len__(self) -> int:
        """Returns the number of datasets in this DsSs"""
        out = len(self.__data)
        self.logger.log_micro_function("ds_len", "return", retval=out)
        return out
