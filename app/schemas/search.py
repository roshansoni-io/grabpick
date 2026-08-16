from typing import List, Optional

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single ranked person returned by a search."""

    person_id: str
    name: Optional[str] = None
    similarity: float = Field(0.0, ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    limit: int
    results: List[SearchResult]


class PersonImage(BaseModel):
    image: str
    original_url: Optional[str] = None


class PersonImagesResponse(BaseModel):
    person_id: str
    name: Optional[str] = None
    images: List[PersonImage]
