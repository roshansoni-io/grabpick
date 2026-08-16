from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from ..exceptions import ModelLoadError
from ..preprocessing import normalize_image, to_chw
from ..session import load_session
from ..types import Detection
from ...utils.logger import logger

_CONFIGS = {
    6: (3, [8, 16, 32], 2, False),
    9: (3, [8, 16, 32], 2, True),
    10: (5, [8, 16, 32, 64, 128], 1, False),
    15: (5, [8, 16, 32, 64, 128], 1, True),
}


class FaceDetector:
    """SCRFD-based face detector running on ONNX Runtime."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        num_threads: int = 0,
        providers: Optional[List[str]] = None,
        session: Optional[ort.InferenceSession] = None,
    ):
        self.session = session or load_session(model_path, num_threads, providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.conf_thresh = confidence_threshold
        self.nms_thresh = nms_threshold

        if len(self.output_names) not in _CONFIGS:
            raise ModelLoadError(
                f"Unsupported model output count: {len(self.output_names)}"
            )
        self.feat_count, self.strides, self.anchors, self.has_kp = _CONFIGS[
            len(self.output_names)
        ]
        self._centers_cache: dict = {}

    def _centers(self, input_size: Tuple[int, int]) -> List[np.ndarray]:
        """Anchor centers for each stride, cached per input size."""
        if input_size not in self._centers_cache:
            h, w = input_size
            cache = []
            for stride in self.strides:
                gy, gx = np.mgrid[: h // stride, : w // stride]
                centers = (np.stack((gx, gy), axis=-1).astype(np.float32) * stride).reshape(-1, 2)
                if self.anchors > 1:
                    centers = np.repeat(centers, self.anchors, axis=0)
                cache.append(centers)
            self._centers_cache[input_size] = cache
        return self._centers_cache[input_size]

    def detect(
        self,
        image: np.ndarray,
        input_size: Tuple[int, int] = (640, 640),
    ) -> List[Detection]:
        """Detect faces and return bounding boxes with optional landmarks."""
        if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
            raise ValueError(f"Expected a 3D image, got shape {image.shape}")
        centers = self._centers(input_size)

        h0, w0 = image.shape[:2]
        scale = min(input_size[1] / h0, input_size[0] / w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            0,
            input_size[1] - new_h,
            0,
            input_size[0] - new_w,
            cv2.BORDER_CONSTANT,
        )
        blob = to_chw(normalize_image(padded))

        try:
            outputs = self.session.run(self.output_names, {self.input_name: blob})
        except Exception as exc:
            logger.warning(f"Detection inference failed: {exc}")
            return []

        scores_list, boxes_list, kps_list = [], [], []
        for i, stride in enumerate(self.strides):
            scores = outputs[i].ravel()
            mask = scores >= self.conf_thresh
            if not np.any(mask):
                continue
            scores_list.append(scores[mask])

            c = centers[i][mask]
            dist = outputs[i + self.feat_count].reshape(-1, 4)[mask] * stride
            boxes_list.append(
                np.stack(
                    [
                        c[:, 0] - dist[:, 0],
                        c[:, 1] - dist[:, 1],
                        c[:, 0] + dist[:, 2],
                        c[:, 1] + dist[:, 3],
                    ],
                    axis=1,
                )
            )

            if self.has_kp:
                kps = outputs[i + self.feat_count * 2].reshape(-1, 5, 2)[mask] * stride
                kps[..., 0] += c[:, 0:1]
                kps[..., 1] += c[:, 1:2]
                kps_list.append(kps)

        if not scores_list:
            return []

        scores = np.concatenate(scores_list)
        boxes = np.concatenate(boxes_list) / scale
        kps_raw = np.concatenate(kps_list) / scale if self.has_kp else None

        boxes_wh = boxes.copy()
        boxes_wh[:, 2] -= boxes_wh[:, 0]
        boxes_wh[:, 3] -= boxes_wh[:, 1]
        indices = cv2.dnn.NMSBoxes(
            boxes_wh.tolist(), scores.tolist(), self.conf_thresh, self.nms_thresh
        )
        if len(indices) == 0:
            return []
        indices = np.asarray(indices).flatten()

        results = []
        for idx in indices:
            landmarks = None
            if kps_raw is not None:
                landmarks = [tuple(map(float, kp)) for kp in kps_raw[idx]]
            results.append(
                Detection(
                    bbox=tuple(map(int, boxes[idx])),
                    score=float(scores[idx]),
                    landmarks=landmarks,
                )
            )
        return results