from enum import IntEnum

from cluster_classification.classification_logger import ClassificationLogger

logger = ClassificationLogger()

class SignalType(IntEnum):
    """Types of signal an individual event may be part of.
    
    Extends: `enum.IntEnum`

    Values:
    - `BACKGROUND` for background events;
    - `HADRONIC` for hadronic events.
    """

    BACKGROUND = 1,
    HADRONIC = 2,

logger.log_debug('Loaded Signal Types')
