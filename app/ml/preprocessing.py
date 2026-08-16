from typing import List, Optional, Tuple

import cv2
import numpy as np

from .types import Detection

ALIGNMENT_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

FACE_SIZE = 112


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = FACE_SIZE,
) -> np.ndarray:
    """Align a face using 5-point landmarks."""
    if landmarks is None or landmarks.shape[0] < 5:
        raise ValueError(
            f"Expected 5 landmarks for alignment, got {landmarks.shape}"
        )
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks[:5].astype(np.float32), ALIGNMENT_TEMPLATE
    )
    if matrix is None:
        raise ValueError("Could not estimate affine transformation from landmarks")
    return cv2.warpAffine(
        image,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_face(
    image: np.ndarray,
    bbox: np.ndarray,
    output_size: int = FACE_SIZE,
) -> np.ndarray:
    """Crop and resize a face from a bounding box (x1, y1, x2, y2)."""
    if len(bbox) < 4:
        raise ValueError(f"Invalid bounding box with shape {bbox.shape}")
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.clip(bbox[:4].astype(np.int32), 0, [w, h, w, h])
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    return cv2.resize(image[y1:y2, x1:x2], (output_size, output_size))


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (127.5, 127.5, 127.5),
    std: float = 128.0,
) -> np.ndarray:
    """Standardize image pixels to float32: (image - mean) / std."""
    return (image.astype(np.float32) - np.array(mean, np.float32)) / np.float32(std)


def to_chw(image: np.ndarray) -> np.ndarray:
    """Convert an HWC image to a batched NCHW array."""
    if image.ndim == 3:
        return image.transpose(2, 0, 1)[None]
    if image.ndim == 4:
        return image.transpose(0, 3, 1, 2)
    raise ValueError(f"Image must be 3D or 4D, got {image.ndim}D")


def annotate(
    image: np.ndarray,
    detections: List[Detection],
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw bounding boxes, landmarks, and labels on an image."""
    output = image.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det.bbox)
        color = (0, 255, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = labels[i] if labels and i < len(labels) else f"{det.score:.2f}"
        cv2.putText(
            output, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        if det.landmarks:
            for x, y in det.landmarks:
                cv2.circle(output, (int(x), int(y)), 2, (255, 0, 0), -1)
    return output