import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(slots=True)
class Detection:
    bbox: Tuple[int, int, int, int]
    score: float
    landmarks: Optional[list[Tuple[float, float]]] = None


@dataclass(slots=True)
class RecognitionResult:
    detection: Optional["Detection"] = None
    person_id: str = "unknown"
    similarity: float = 0.0
    embedding: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return self.person_id