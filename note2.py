from dataclasses import dataclass
from enum import Enum
from typing import Tuple

class NoteTechnique(Enum):
    STRUM = "strum"
    HOPO = "hopo"
    TAP = "tap"


@dataclass
class Note:
    # time remaining before the next note
    time: float

    # Five fret indicators
    frets: Tuple[bool, bool, bool, bool, bool]

    # Independent sustain duration for each fret
    durations: Tuple[float, float, float, float, float]

    # How the note must be played
    technique: NoteTechnique

    # Whether this is an open note
    is_open: bool = False
