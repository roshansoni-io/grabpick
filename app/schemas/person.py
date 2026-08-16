from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Person(BaseModel):
    """A known identity/person with accumulated face embeddings."""

    person_id: str
    name: str
    embedding_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PersonCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class PersonList(BaseModel):
    people: List[Person]
