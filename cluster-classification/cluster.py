
from enum import Enum

class ClusterType(Enum):
    """
    The `ClusterType` enum exists to mark the type of event under consideration.
    """

    BACKGROUND = 1,
    SIGNAL_HADRONIC = 2,
