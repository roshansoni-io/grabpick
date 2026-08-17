"""Person (identity) management service."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .. import ml
from ..database import (
    DatabaseError,
    delete_identity,
    get_identity,
    list_identities,
    save_identity,
)
from ..exceptions import ImageDecodeError, ProcessingError
from ..schemas.person import Person
from ..utils import storage
from ..utils.logger import logger


def _to_person(item: dict) -> Person:
    return Person(
        person_id=item["person_id"],
        name=item["name"],
        embedding_count=item["embedding_count"],
        created_at=item.get("created_at"),
        updated_at=item.get("updated_at"),
    )

def _face_area(detection) -> float:
    """Return the area of a face bounding box."""
    x1, y1, x2, y2 = detection.bbox

    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def list_people() -> List[Person]:
    try:
        items = list_identities()
    except DatabaseError as exc:
        logger.error("list_people failed: %s", exc)
        raise
    return [_to_person(item) for item in items]


def get_person(person_id: str) -> Optional[Person]:
    try:
        item = get_identity(person_id)
    except DatabaseError as exc:
        logger.error("get_person failed: %s", exc)
        raise
    return _to_person(item) if item is not None else None


def create_person(
    name: str,
    embedding: np.ndarray,
    source_image: str,
) -> Person:
    """Create (or append to) an identity."""
    try:
        person_id = save_identity(
            name=name,
            embedding=embedding,
            source_image=source_image,
        )
    except DatabaseError as exc:
        logger.error("create_person failed: %s", exc)
        raise
    person = get_person(person_id)
    if person is None:
        raise DatabaseError("Identity was not found after creation")
    return person


def enroll_person(name: str, data: bytes, filename: str) -> Person:
    """Enroll a person using a reference image.

    The image is decoded, valid face detections are filtered, and the
    largest valid face is selected for embedding. The original image is
    saved only after successful face processing. If database enrollment
    fails, the saved image is removed to avoid orphaned files.
    """
    name = name.strip()

    if not name:
        raise ProcessingError("Person name cannot be empty")

    if not data:
        raise ProcessingError("Submitted image is empty")

    # Decode the uploaded image.
    try:
        image = storage.decode_image(data)
    except ImageDecodeError as exc:
        raise ProcessingError("Could not decode image") from exc

    # Detect faces and generate embeddings.
    try:
        results = ml.identify(image)
    except Exception as exc:
        raise ProcessingError("Face processing failed") from exc

    # Keep only detections with a valid face, embedding, and confidence.
    valid = [
        result
        for result in results
        if (
            result.detection is not None
            and result.embedding is not None
            and result.detection.score >= 0.5
        )
    ]

    if not valid:
        raise ProcessingError(
            "No valid face found in the submitted reference image"
        )

    # Select the largest detected face.
    best = max(
        valid,
        key=lambda result: _face_area(result.detection),
    )

    # Save the original image.
    original, created = storage.save_original(data, storage.safe_filename(filename))

    try:
        # Store the person and embedding in the database.
        return create_person(
            name=name,
            embedding=best.embedding,
            source_image=str(original),
        )
    except Exception:
        if created:
            try:
                storage.delete_original(original)
            except Exception:
                pass

        raise


def delete_person(person_id: str) -> bool:
    """Delete an identity and all of its embeddings."""
    if get_person(person_id) is None:
        return False
    try:
        delete_identity(person_id)
    except DatabaseError as exc:
        logger.error("delete_person failed: %s", exc)
        raise
    return True
