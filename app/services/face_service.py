"""Face detection/embedding: thin wrappers around the ML public API.

Identity resolution against the database happens here via pgvector, so the
ML layer stays a pure detect+embed pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .. import ml
from ..database import match_identity, get_identity_names
from ..ml.types import Detection, RecognitionResult
from ..utils.logger import logger


def detect_faces(
    image: np.ndarray,
    confidence_threshold: Optional[float] = None,
) -> List[Detection]:
    """Detect faces in an image without matching."""
    return ml.face_locations(
        image, confidence_threshold=confidence_threshold
    )


def run_identify(
    image: np.ndarray,
    recognize_threshold: Optional[float] = None,
) -> List[RecognitionResult]:
    """Detect, embed, and resolve identities against the database.

    Each detected face is embedded and matched with pgvector
    (``match_identity``); faces below ``recognize_threshold`` are marked
    ``"unknown"``.
    """
    threshold = recognize_threshold or ml.settings.recognize_threshold
    results = ml.identify(image)
    resolved: List[RecognitionResult] = []
    for result in results:
        if result.embedding is not None:
            person_id, similarity = match_identity(result.embedding, threshold=threshold)
            result.person_id = person_id
            result.similarity = similarity
        resolved.append(result)
    return resolved


def embed_face(
    image: np.ndarray,
    bbox: tuple,
) -> Optional[np.ndarray]:
    """Embed a single face given a detection bbox."""
    det = Detection(bbox=tuple(map(int, bbox)), score=1.0)
    encodings = ml.face_encodings(image, [det])
    return encodings[0] if encodings else None


def resolve_names(person_ids: List[str]) -> Dict[str, str]:
    """Map person_ids to display names using the DB (single query)."""
    try:
        names = get_identity_names(person_ids)
    except Exception as exc:
        logger.error("resolve_names failed: %s", exc)
        return {}
    return {pid: names.get(pid, pid) for pid in person_ids}
