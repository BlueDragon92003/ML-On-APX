
from enum import IntEnum

class SignalType(IntEnum):
    """Types of signal an individual event may be part of.
    
    Extends: `enum.IntEnum`

    Values:
    - `BACKGROUND` for background events;
    - `HADRONIC` for hadronic events.
    """

    BACKGROUND = 1,
    HADRONIC = 2,
