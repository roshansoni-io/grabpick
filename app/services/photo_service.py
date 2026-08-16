"""Photo upload and processing service."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from .. import ml
from ..exceptions import ProcessingError
from ..ml.types import Detection, RecognitionResult
from ..schemas.face import Face, FaceMatch
from ..schemas.photo import Photo, PhotoUploadResponse
from ..utils import storage
from ..utils.logger import logger
from .face_service import resolve_names, run_identify


def _detections_to_faces(
    image: np.ndarray,
    results: List[RecognitionResult],
) -> Tuple[List[Face], List[FaceMatch]]:
    faces: List[Face] = []
    matches: List[FaceMatch] = []
    known_ids = [r.person_id for r in results if r.person_id != "unknown"]
    names = resolve_names(list(dict.fromkeys(known_ids)))

    for result in results:
        det = result.detection
        if det is None:
            continue
        crop_url = storage.thumbnail_url(storage.save_face_crop(image, det.bbox))
        faces.append(
            Face(
                bbox=tuple(map(int, det.bbox)),
                score=det.score,
                landmarks=[tuple(lm) for lm in det.landmarks] if det.landmarks else None,
                thumbnail_url=crop_url,
            )
        )
        if result.person_id != "unknown":
            matches.append(
                FaceMatch(
                    bbox=tuple(map(int, det.bbox)),
                    score=det.score,
                    person_id=result.person_id,
                    name=names.get(result.person_id),
                    similarity=result.similarity,
                )
            )
    return faces, matches


def process_upload(data: bytes, filename: str) -> PhotoUploadResponse:
    """Save an uploaded photo, detect/identify faces, and return results."""
    original = storage.save_original(data, filename)

    try:
        image = ml.load_image(original)
    except Exception as exc:
        logger.error("process_upload: could not decode image: %s", exc)
        raise ProcessingError(f"Could not decode uploaded image: {exc}") from exc

    try:
        results = run_identify(image)
    except Exception as exc:
        logger.error("process_upload: recognition failed: %s", exc)
        raise ProcessingError(f"Face recognition failed: {exc}") from exc

    faces, matches = _detections_to_faces(image, results)
    thumb_url = storage.thumbnail_url(storage.save_thumbnail(image))

    photo = Photo(
        id=original.stem,
        filename=filename,
        uploaded_at=datetime.utcnow(),
        original_url=storage.original_url(original),
        thumbnail_url=thumb_url,
        face_count=len(faces),
    )
    return PhotoUploadResponse(photo=photo, faces=faces, matches=matches)
