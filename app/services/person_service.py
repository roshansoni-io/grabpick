"""Person (identity) management service."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .. import ml
from ..database import (
    DatabaseError,
    delete_identity,
    list_identities,
    save_identity,
)
from ..exceptions import NotFoundError, ProcessingError, ServiceError
from ..schemas.person import Person
from ..utils import storage
from ..utils.logger import logger
from .face_service import reload_database


def _to_person(item: dict) -> Person:
    return Person(
        person_id=item["person_id"],
        name=item["name"],
        embedding_count=item["embedding_count"],
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
    )


def list_people() -> List[Person]:
    try:
        items = list_identities()
    except DatabaseError as exc:
        logger.error("list_people failed: %s", exc)
        raise
    return [_to_person(item) for item in items]


def get_person(person_id: str) -> Optional[Person]:
    for person in list_people():
        if person.person_id == person_id:
            return person
    return None


def create_person(
    name: str,
    embedding: np.ndarray,
    source_image: str,
) -> Person:
    """Create (or append to) an identity and rebuild the matcher."""
    try:
        person_id = save_identity(
            name=name,
            embedding=embedding,
            source_image=source_image,
        )
    except DatabaseError as exc:
        logger.error("create_person failed: %s", exc)
        raise
    reload_database()
    person = get_person(person_id)
    if person is None:
        raise DatabaseError("Identity was not found after creation")
    return person


def enroll_person(name: str, data: bytes, filename: str) -> Person:
    """Enroll a new person from an uploaded reference image.

    Detects the largest face, embeds it, and registers the identity with
    the database, then rebuilds the in-memory matcher.
    """
    try:
        image = storage.decode_image(data)
    except Exception as exc:
        raise ProcessingError(f"Could not decode image: {exc}")

    results = ml.identify(image)
    if not results:
        raise ProcessingError("No face found in the submitted reference image")

    best = max(results, key=lambda r: r.detection.score if r.detection else 0.0)
    if best.embedding is None or best.detection is None:
        raise ProcessingError("Could not extract a face embedding")

    original = storage.save_original(data, filename)
    person = create_person(name, best.embedding, str(original))
    return person


def delete_person(person_id: str) -> bool:
    """Delete an identity and rebuild the matcher."""
    if get_person(person_id) is None:
        return False
    try:
        delete_identity(person_id)
    except DatabaseError as exc:
        logger.error("delete_person failed: %s", exc)
        raise
    reload_database()
    return True
