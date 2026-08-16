"""Image search service: embed a query image and find nearest people."""

from __future__ import annotations

from typing import List

import numpy as np

from .. import ml
from ..database import DatabaseError, search_embeddings
from ..exceptions import ProcessingError
from ..schemas.search import SearchResponse, SearchResult
from ..utils import storage
from ..utils.logger import logger
from .face_service import resolve_names


def search_image_from_bytes(
    data: bytes,
    limit: int = 10,
    recognize_threshold: float = 0.45,
) -> SearchResponse:
    try:
        image = storage.decode_image(data)
    except Exception as exc:
        raise ProcessingError(f"Could not decode query image: {exc}")
    return search_image(image, limit, recognize_threshold)


def search_image(
    image: np.ndarray,
    limit: int = 10,
    recognize_threshold: float = 0.45,
) -> SearchResponse:
    """Embed the query image and rank the nearest people by similarity."""
    encodings = ml.face_encodings(image)
    if not encodings:
        return SearchResponse(limit=limit, results=[])

    from collections import OrderedDict

    query = np.mean(encodings, axis=0)
    try:
        ranked = search_embeddings(query, limit=limit)
    except DatabaseError as exc:
        logger.error("search_image failed: %s", exc)
        raise

    best_by_person: "OrderedDict[str, float]" = OrderedDict()
    for pid, score in ranked:
        if best_by_person.get(pid, 0.0) < score:
            best_by_person[pid] = score

    names = resolve_names(list(best_by_person.keys()))
    results = [
        SearchResult(person_id=pid, name=names.get(pid), similarity=score)
        for pid, score in best_by_person.items()
        if score >= recognize_threshold
    ]
    return SearchResponse(limit=limit, results=results)
