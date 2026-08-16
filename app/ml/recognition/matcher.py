from typing import Dict, List, Tuple, Union

import numpy as np

from ...utils.logger import logger

Match = Tuple[str, float]


class FaceMatcher:
    """Matches query embeddings against a database using cosine similarity."""

    def __init__(self, database: Dict[str, np.ndarray], threshold: float = 0.45):
        self.threshold = threshold
        ids, templates = [], []
        for person_id, embs in database.items():
            if embs.size == 0:
                continue
            ids.append(person_id)
            templates.append(embs)

        if not templates:
            self.matrix = None
            self.index_map = []
            logger.warning("FaceMatcher initialized with an empty database")
            return

        self.index_map = [
            pid for pid, arr in zip(ids, templates) for _ in range(len(arr))
        ]
        self.matrix = np.vstack(templates).astype(np.float32)

    def match(self, embeddings: np.ndarray) -> Union[Match, List[Match]]:
        """Match one or more embeddings and return (person_id, score)."""
        if self.matrix is None:
            unknown: Match = ("unknown", 0.0)
            return unknown if embeddings.ndim == 1 else [unknown] * len(embeddings)

        is_single = embeddings.ndim == 1
        scores = (embeddings[None, :] if is_single else embeddings) @ self.matrix.T
        best = np.argmax(scores, axis=1)

        results = []
        for row, idx in zip(scores, best):
            score = float(row[idx])
            results.append(
                (self.index_map[idx], score)
                if score >= self.threshold
                else ("unknown", score)
            )
        return results[0] if is_single else results