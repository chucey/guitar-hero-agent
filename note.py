from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Note:
    """One chart event. frets is a 5-element tuple in G,R,Y,B,O order."""
    time: float
    frets: Tuple[int, int, int, int, int]
    duration: float = 0.0

    def __post_init__(self):
        if len(self.frets) != 5 or any(x not in (0, 1) for x in self.frets):
            raise ValueError("frets must be five binary values")
        if not any(self.frets):
            raise ValueError("This starter environment does not support open notes")
        if self.time < 0 or self.duration < 0:
            raise ValueError("time and duration must be non-negative")
