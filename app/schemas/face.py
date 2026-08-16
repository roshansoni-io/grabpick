from typing import List, Optional, Tuple

from pydantic import BaseModel


class Face(BaseModel):
    """A single detected face within an image."""

    bbox: Tuple[int, int, int, int]
    score: float
    landmarks: Optional[List[Tuple[float, float]]] = None
    thumbnail_url: Optional[str] = None


class FaceMatch(BaseModel):
    """A face matched to a known person."""

    bbox: Tuple[int, int, int, int]
    score: float
    person_id: str
    name: Optional[str] = None
    similarity: float = 0.0
