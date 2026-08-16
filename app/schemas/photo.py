from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from .face import Face, FaceMatch


class Photo(BaseModel):
    """Metadata for an uploaded/processed photo."""

    id: str
    filename: str
    uploaded_at: datetime
    original_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    face_count: int = 0


class PhotoUploadResponse(BaseModel):
    """Result of uploading and processing a photo."""

    photo: Photo
    faces: List[Face]
    matches: List[FaceMatch]
