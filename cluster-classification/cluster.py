
from enum import IntEnum

class ClusterType(IntEnum):
    """
    The `ClusterType` enum exists to mark the type of event under consideration.
    """

    BACKGROUND = 1,
    SIGNAL_HADRONIC = 2,
