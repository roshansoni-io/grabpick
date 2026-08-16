from typing import List, Optional, Tuple

import numpy as np

from ...config import Settings
from ...utils.logger import logger
from ..detection.detector import FaceDetector
from ..embedding.embedder import FaceEmbedder
from ..preprocessing import align_face, crop_face
from ..types import Detection, RecognitionResult
from .matcher import FaceMatcher


def extract_faces(
    image: np.ndarray,
    detections: List[Detection],
) -> Tuple[List[np.ndarray], List[Detection]]:
    """Align or crop each detection; invalid ones are skipped."""
    faces, valid = [], []
    for det in detections:
        try:
            if det.landmarks is not None:
                face = align_face(image, np.asarray(det.landmarks, np.float32))
            else:
                face = crop_face(image, np.asarray(det.bbox, np.float32))
        except ValueError:
            logger.warning("Skipping detection: face alignment/crop failed")
            continue
        faces.append(face)
        valid.append(det)
    return faces, valid


class RecognitionPipeline:
    """End-to-end face recognition: detect, align, embed, match."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: Optional[FaceEmbedder],
        matcher: Optional[FaceMatcher] = None,
    ):
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher

    def update_settings(self, settings: Settings) -> None:
        if self.detector:
            self.detector.conf_thresh = settings.confidence_threshold
            self.detector.nms_thresh = settings.nms_threshold
        if self.matcher:
            self.matcher.threshold = settings.recognize_threshold

    def run(
        self,
        image: np.ndarray,
        input_size: Tuple[int, int] = (640, 640),
    ) -> List[RecognitionResult]:
        detections = self.detector.detect(image, input_size)
        if not detections:
            return []

        faces, valid = extract_faces(image, detections)
        if not faces:
            return []

        if self.embedder is None:
            return [RecognitionResult(detection=d) for d in valid]

        try:
            embeddings = self.embedder.embed(faces)
        except Exception:
            logger.error("Embedding failed")
            return [RecognitionResult(detection=d) for d in valid]

        matches = self._match(embeddings)

        return [
            RecognitionResult(
                detection=det,
                person_id=person_id,
                similarity=score,
                embedding=emb,
            )
            for det, emb, (person_id, score) in zip(valid, embeddings, matches)
        ]

    def _match(self, embeddings: np.ndarray) -> List[Tuple[str, float]]:
        if self.matcher is None:
            return [("unknown", 0.0)] * len(embeddings)
        results = self.matcher.match(embeddings)
        return results if isinstance(results, list) else [results]