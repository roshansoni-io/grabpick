"""Photo upload and processing service."""

from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

import numpy as np

from ..exceptions import ProcessingError
from ..database import DatabaseError, save_image_metadata
from ..ml.types import RecognitionResult
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
    """Validate and decode an uploaded photo, save it, then detect/identify faces.

    The image is decoded (and validated) *before* being persisted so that
    garbage or oversized files never land on disk.
    """
    image = storage.decode_image(data)
    safe_name = storage.safe_filename(filename)
    original, _ = storage.save_original(data, safe_name)

    try:
        results = run_identify(image)
    except Exception as exc:
        logger.error("process_upload: recognition failed: %s", exc)
        raise ProcessingError(f"Face recognition failed: {exc}") from exc

    faces, matches = _detections_to_faces(image, results)
    thumb_url = storage.thumbnail_url(storage.save_thumbnail(image))

    try:
        people = [
            {"person_id": m.person_id, "name": m.name} for m in matches
        ]
        save_image_metadata(original.name, str(original), people)
    except DatabaseError as exc:
        logger.warning("process_upload: could not record image metadata: %s", exc)

    photo = Photo(
        id=original.stem,
        filename=safe_name,
        uploaded_at=datetime.utcnow(),
        original_url=storage.original_url(original),
        thumbnail_url=thumb_url,
        face_count=len(faces),
    )
    return PhotoUploadResponse(photo=photo, faces=faces, matches=matches)
