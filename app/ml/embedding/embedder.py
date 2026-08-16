from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from ..exceptions import EmbeddingError
from ..preprocessing import normalize_image
from ..session import load_session
from ...utils.logger import logger


class FaceEmbedder:
    """ONNX face embedding extractor producing L2-normalized vectors."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_threads: int = 2,
        providers: Optional[List[str]] = None,
        session: Optional[ort.InferenceSession] = None,
    ):
        self.session = session or load_session(model_path, num_threads, providers)
        meta = self.session.get_inputs()[0]
        self.input_name = meta.name
        self.input_size = (meta.shape[3], meta.shape[2])
        self.out_dim = self.session.get_outputs()[0].shape[1]

    def embed(
        self,
        faces: List[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract L2-normalized embeddings for a batch of face crops."""
        if not faces:
            return np.empty((0, self.out_dim), dtype=np.float32)

        try:
            batches = []
            for start in range(0, len(faces), batch_size):
                chunk = np.stack(
                    [
                        normalize_image(cv2.resize(face, self.input_size)).transpose(2, 0, 1)
                        for face in faces[start : start + batch_size]
                    ]
                )
                batches.append(self.session.run(None, {self.input_name: chunk})[0])

            embeddings = np.concatenate(batches)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / np.maximum(norms, 1e-6)
        except Exception as exc:
            logger.error(f"Embedding failed: {exc}")
            raise EmbeddingError(f"Embedding failed: {exc}") from exc