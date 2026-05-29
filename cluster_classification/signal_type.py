from enum import IntEnum

from cluster_classification.classification_logger import CleverLogger

logger = CleverLogger(__name__)

logger.log_start_load_module()

class SignalType(IntEnum):
    """Types of signal an individual event may be part of.

    Extends: `enum.IntEnum`

    Values:
    - `BACKGROUND` for background events;
    - `HADRONIC` for hadronic events.
    """

    BACKGROUND = (1,)
    HADRONIC = (2,)


logger.log_end_load_module()
