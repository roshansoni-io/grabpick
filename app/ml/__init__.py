"""Face recognition API: detection, embedding, and identification."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ..config import settings
from ..utils.logger import logger
from .detection.detector import FaceDetector
from .embedding.embedder import FaceEmbedder
from .exceptions import FaceRecognitionError
from .preprocessing import annotate
from .recognition.matcher import FaceMatcher
from .recognition.pipeline import RecognitionPipeline, extract_faces
from .types import Detection, RecognitionResult

__version__ = "0.1.0"

_DETECTOR: Optional[FaceDetector] = None
_EMBEDDER: Optional[FaceEmbedder] = None
_PIPELINE: Optional[RecognitionPipeline] = None
_DETECTOR_KEY: Optional[Tuple] = None
_EMBEDDER_KEY: Optional[Tuple] = None
_LOCK = threading.RLock()


def _detector() -> FaceDetector:
    global _DETECTOR, _DETECTOR_KEY
    key = (
        str(settings.detector_model),
        settings.providers,
        settings.num_threads,
        settings.confidence_threshold,
        settings.nms_threshold,
        settings.input_size,
    )
    with _LOCK:
        if _DETECTOR is None or key != _DETECTOR_KEY:
            _DETECTOR = FaceDetector(
                model_path=str(settings.detector_model),
                providers=settings.providers,
                num_threads=settings.num_threads,
                confidence_threshold=settings.confidence_threshold,
                nms_threshold=settings.nms_threshold,
            )
            _DETECTOR_KEY = key
        else:
            _DETECTOR.conf_thresh = settings.confidence_threshold
            _DETECTOR.nms_thresh = settings.nms_threshold
    return _DETECTOR


def _embedder() -> FaceEmbedder:
    global _EMBEDDER, _EMBEDDER_KEY
    key = (str(settings.embedding_model), settings.providers, settings.num_threads)
    with _LOCK:
        if _EMBEDDER is None or key != _EMBEDDER_KEY:
            _EMBEDDER = FaceEmbedder(
                model_path=str(settings.embedding_model),
                providers=settings.providers,
                num_threads=settings.num_threads,
            )
            _EMBEDDER_KEY = key
    return _EMBEDDER


def _pipeline() -> RecognitionPipeline:
    global _PIPELINE
    detector, embedder = _detector(), _embedder()
    with _LOCK:
        if (
            _PIPELINE is None
            or _PIPELINE.detector is not detector
            or _PIPELINE.embedder is not embedder
        ):
            _PIPELINE = RecognitionPipeline(detector=detector, embedder=embedder)
        _PIPELINE.update_settings(settings)
    return _PIPELINE


def reset_pipeline() -> None:
    global _DETECTOR, _EMBEDDER, _PIPELINE, _DETECTOR_KEY, _EMBEDDER_KEY
    with _LOCK:
        _DETECTOR = None
        _EMBEDDER = None
        _PIPELINE = None
        _DETECTOR_KEY = None
        _EMBEDDER_KEY = None


def load_image(path: Union[str, Path], mode: str = "RGB") -> np.ndarray:
    """Load an image as a numpy array (H, W, C)."""
    try:
        with Image.open(path) as image:
            if mode:
                image = image.convert(mode)
            return np.asarray(image)
    except Exception as exc:
        raise FaceRecognitionError(f"Failed to load image: {path}") from exc


def face_locations(
    image: np.ndarray,
    input_size: Optional[Tuple[int, int]] = None,
    confidence_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
) -> List[Detection]:
    """Detect faces in an image and return bounding boxes with landmarks."""
    with _LOCK:
        if input_size is not None:
            settings.input_size = input_size
        if confidence_threshold is not None:
            settings.confidence_threshold = confidence_threshold
        if nms_threshold is not None:
            settings.nms_threshold = nms_threshold
    return _pipeline().detector.detect(image, settings.input_size)


def _normalize_locations(
    locations: Union[Detection, Tuple[int, int, int, int], List],
) -> List[Detection]:
    if isinstance(locations, Detection) or (
        isinstance(locations, (list, tuple)) and len(locations) == 4
    ):
        locations = [locations]

    detections = []
    for loc in locations:
        if isinstance(loc, Detection):
            detections.append(loc)
        elif len(loc) == 4:
            detections.append(Detection(bbox=tuple(map(int, loc)), score=1.0))
        else:
            logger.warning(f"Skipping invalid location: {loc}")
    return detections


def face_encodings(
    image: np.ndarray,
    known_face_locations: Optional[
        Union[Detection, Tuple[int, int, int, int], List]
    ] = None,
    input_size: Optional[Tuple[int, int]] = None,
    num_threads: Optional[int] = None,
) -> List[np.ndarray]:
    """Generate normalized face embeddings. Detects faces if no locations given."""
    with _LOCK:
        if input_size is not None:
            settings.input_size = input_size
        if num_threads is not None:
            settings.num_threads = num_threads

    pipeline = _pipeline()
    if pipeline.embedder is None:
        raise FaceRecognitionError("Face embedder is not initialized")

    if known_face_locations is not None:
        detections = _normalize_locations(known_face_locations)
    else:
        detections = pipeline.detector.detect(image, settings.input_size)

    faces, _ = extract_faces(image, detections)
    if not faces:
        return []
    return list(pipeline.embedder.embed(faces))


def set_database(
    database: Dict[str, np.ndarray],
    threshold: Optional[float] = None,
) -> None:
    """Set the identity database used by the recognition pipeline."""
    with _LOCK:
        if threshold is not None:
            settings.recognize_threshold = threshold
        _pipeline().matcher = FaceMatcher(
            database=database, threshold=settings.recognize_threshold
        )


def identify(
    image: np.ndarray,
    input_size: Optional[Tuple[int, int]] = None,
    confidence_threshold: Optional[float] = None,
    recognize_threshold: Optional[float] = None,
) -> List[RecognitionResult]:
    """Run end-to-end recognition: detect, align, embed, match."""
    with _LOCK:
        if input_size is not None:
            settings.input_size = input_size
        if confidence_threshold is not None:
            settings.confidence_threshold = confidence_threshold
        if recognize_threshold is not None:
            settings.recognize_threshold = recognize_threshold
        pipeline = _pipeline()
    return pipeline.run(image, settings.input_size)


def face_similarity(
    encodings: Union[List[np.ndarray], np.ndarray],
    target: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between encodings and a target embedding."""
    encodings_arr = np.asarray(encodings, dtype=np.float32)
    if encodings_arr.ndim == 1:
        encodings_arr = encodings_arr[None]
    return encodings_arr @ np.asarray(target, dtype=np.float32)


def face_distance(
    encodings: Union[List[np.ndarray], np.ndarray],
    target: np.ndarray,
) -> np.ndarray:
    """Euclidean distance between encodings and a target embedding."""
    encodings_arr = np.asarray(encodings, dtype=np.float32)
    if encodings_arr.ndim == 1:
        encodings_arr = encodings_arr[None]
    return np.linalg.norm(
        encodings_arr - np.asarray(target, dtype=np.float32), axis=1
    )


def compare_faces(
    known_encodings: Union[List[np.ndarray], np.ndarray],
    candidate: np.ndarray,
    tolerance: float = 0.55,
) -> List[bool]:
    """Check whether a candidate embedding matches any known encoding."""
    return (face_similarity(known_encodings, candidate) >= (1.0 - tolerance)).tolist()


def draw_faces(
    image: np.ndarray,
    results: Union[List[Detection], List[RecognitionResult]],
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw detections or recognition results on an image."""
    if not results:
        return image.copy()
    if isinstance(results[0], RecognitionResult):
        detections = [r.detection for r in results]
        if labels is None:
            labels = [f"{r.person_id}({r.similarity:.2f})" for r in results]
    else:
        detections = results
    return annotate(image, detections, labels)


__all__ = [
    "FaceDetector",
    "FaceEmbedder",
    "FaceMatcher",
    "RecognitionPipeline",
    "compare_faces",
    "draw_faces",
    "face_distance",
    "face_encodings",
    "face_locations",
    "face_similarity",
    "identify",
    "load_image",
    "reset_pipeline",
    "set_database",
    "settings",
]