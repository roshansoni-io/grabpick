from typing import List, Optional, Tuple

import numpy as np

from ...config import Settings
from ...utils.logger import logger
from ..detection.detector import FaceDetector
from ..embedding.embedder import FaceEmbedder
from ..preprocessing import align_face, crop_face
from ..types import Detection, RecognitionResult


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
    """End-to-end face recognition: detect, align, and embed.

    Identity resolution is intentionally *not* done here: the pipeline
    returns detections with embeddings and the service layer matches them
    against the database via pgvector, keeping the ML layer pure.
    """

    def __init__(
        self,
        detector: FaceDetector,
        embedder: Optional[FaceEmbedder],
    ):
        self.detector = detector
        self.embedder = embedder

    def update_settings(self, settings: Settings) -> None:
        if self.detector:
            self.detector.conf_thresh = settings.confidence_threshold
            self.detector.nms_thresh = settings.nms_threshold

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

        return [
            RecognitionResult(
                detection=det,
                embedding=emb,
            )
            for det, emb in zip(valid, embeddings)
        ]
