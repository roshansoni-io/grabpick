"""Image search service: embed a query image and find nearest people."""

from __future__ import annotations

import numpy as np

from .. import ml
from ..config import settings
from ..database import DatabaseError, list_person_images, search_embeddings
from ..exceptions import ProcessingError
from ..schemas.search import PersonImagesResponse, SearchResponse, SearchResult
from ..utils import storage
from ..utils.logger import logger
from .face_service import resolve_names


def search_image_from_bytes(
    data: bytes,
    limit: int = 10,
    recognize_threshold: float = settings.recognize_threshold,
) -> SearchResponse:
    try:
        image = storage.decode_image(data)
    except Exception as exc:
        raise ProcessingError(f"Could not decode query image: {exc}")
    return search_image(image, limit, recognize_threshold)


def search_image(
    image: np.ndarray,
    limit: int = 10,
    recognize_threshold: float = settings.recognize_threshold,
) -> SearchResponse:
    """Embed each query face and rank the nearest people by similarity.

    Faces are scored individually and aggregated per person by taking the
    best (highest) similarity, so a multi-face query image no longer
    produces a blended embedding.
    """
    encodings = ml.face_encodings(image)
    if not encodings:
        return SearchResponse(limit=limit, results=[])

    best_by_person: dict = {}
    try:
        for encoding in encodings:
            ranked = search_embeddings(encoding, limit=limit)
            for pid, score in ranked:
                if score > best_by_person.get(pid, 0.0):
                    best_by_person[pid] = score
    except DatabaseError as exc:
        logger.error("search_image failed: %s", exc)
        raise

    ranked_persons = sorted(
        best_by_person.items(), key=lambda item: item[1], reverse=True
    )[:limit]

    names = resolve_names([pid for pid, _ in ranked_persons])
    results = [
        SearchResult(person_id=pid, name=names.get(pid), similarity=score)
        for pid, score in ranked_persons
        if score >= recognize_threshold
    ]
    return SearchResponse(limit=limit, results=results)


def search_images_for_person(person_id: str) -> PersonImagesResponse:
    """Return every image in which a given person appeared."""
    try:
        images = list_person_images(person_id)
    except DatabaseError as exc:
        logger.error("search_images_for_person failed: %s", exc)
        raise

    names = resolve_names([person_id])
    return PersonImagesResponse(
        person_id=person_id,
        name=names.get(person_id),
        images=[
            {
                "image": image,
                "original_url": storage.original_url(storage.settings.originals_dir / image),
            }
            for image in images
        ],
    )
