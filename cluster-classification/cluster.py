
from enum import Enum

class ClusterType(Enum):
    """
    The `ClusterType` enum exists to mark the type of event under consideration.
    """

    HADRONIC = 1,
    ELECTROMAGNETIC = 2,
