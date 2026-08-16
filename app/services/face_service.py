"""Face detection/recording: thin wrappers around the ML public API."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .. import ml
from ..database import load_database, list_identities
from ..ml.types import Detection, RecognitionResult
from ..utils.logger import logger


def reload_database() -> None:
    """Rebuild the in-memory matcher from the current DB contents."""
    try:
        database = load_database()
    except Exception as exc:
        logger.error("reload_database failed: %s", exc)
        raise
    ml.set_database(database, threshold=ml.settings.recognize_threshold)
    logger.info("Matcher rebuilt with %d identities", len(database))


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
    """Detect, embed, and match faces; returns recognition results."""
    return ml.identify(
        image, recognize_threshold=recognize_threshold
    )


def embed_face(
    image: np.ndarray,
    bbox: tuple,
) -> Optional[np.ndarray]:
    """Embed a single face given a detection bbox."""
    det = Detection(bbox=tuple(map(int, bbox)), score=1.0)
    encodings = ml.face_encodings(image, [det])
    return encodings[0] if encodings else None


def resolve_names(person_ids: List[str]) -> Dict[str, str]:
    """Map person_ids to display names using the DB."""
    if not person_ids:
        return {}
    try:
        identities = list_identities()
    except Exception as exc:
        logger.error("resolve_names failed: %s", exc)
        return {}
    by_id = {item["person_id"]: item["name"] for item in identities}
    return {pid: by_id.get(pid, pid) for pid in person_ids}
