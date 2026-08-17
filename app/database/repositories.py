import uuid
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy import delete, func, select

from ..config import settings
from .connection import session_scope
from .exceptions import DatabaseError
from .models import FaceEmbedding, Identity, Image


def _distance_expr(metric: str, query: np.ndarray):
    """Return the pgvector distance expression and score formula for a metric.

    ``distance`` is the pgvector distance used for ORDER BY (KNN); ``score``
    maps it to a similarity where higher = closer and an exact match = 1.0.
    """
    if metric == "l2":
        distance = FaceEmbedding.vector.l2_distance(query)
        return distance, (1 / (1 + distance))
    if metric == "inner_product":
        distance = FaceEmbedding.vector.max_inner_product(query)
        return distance, -distance
    distance = FaceEmbedding.vector.cosine_distance(query)
    return distance, (1 - distance)


def save_identity(
    name: str,
    embedding: np.ndarray,
    source_image: str,
    person_id: str | None = None,
) -> str:
    """Persist an identity and one embedding row; returns the person_id."""
    person_id = person_id or uuid.uuid4().hex[:8]
    vector = embedding.flatten().astype(np.float32)
    try:
        with session_scope() as session:
            identity = session.get(Identity, person_id)
            if identity is None:
                session.add(Identity(person_id=person_id, name=name))
            else:
                identity.name = name
            session.add(
                FaceEmbedding(person_id=person_id, source_image=source_image, vector=vector)
            )
    except Exception as exc:
        raise DatabaseError(f"Failed to save identity {person_id}: {exc}") from exc
    return person_id


def list_identities() -> List[dict]:
    """Return identity metadata with embedding counts."""
    try:
        with session_scope() as session:
            rows = session.execute(
                select(
                    Identity.person_id,
                    Identity.name,
                    Identity.created_at,
                    Identity.updated_at,
                    func.count(FaceEmbedding.id),
                )
                .outerjoin(FaceEmbedding)
                .group_by(Identity.person_id)
                .order_by(Identity.name)
            ).all()
    except Exception as exc:
        raise DatabaseError(f"Failed to list identities: {exc}") from exc

    return [
        {
            "person_id": person_id,
            "name": name,
            "created_at": created_at,
            "updated_at": updated_at,
            "embedding_count": count,
        }
        for person_id, name, created_at, updated_at, count in rows
    ]


def get_identity(person_id: str) -> dict | None:
    """Return a single identity's metadata with embedding count."""
    try:
        with session_scope() as session:
            row = session.execute(
                select(
                    Identity.person_id,
                    Identity.name,
                    Identity.created_at,
                    Identity.updated_at,
                    func.count(FaceEmbedding.id),
                )
                .outerjoin(FaceEmbedding)
                .where(Identity.person_id == person_id)
                .group_by(Identity.person_id)
            ).first()
    except Exception as exc:
        raise DatabaseError(f"Failed to get identity {person_id}: {exc}") from exc

    if row is None:
        return None
    return {
        "person_id": row.person_id,
        "name": row.name,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "embedding_count": row[4],
    }


def get_identity_names(person_ids: list[str]) -> Dict[str, str]:
    """Map person_ids to their names in a single query."""
    if not person_ids:
        return {}
    try:
        with session_scope() as session:
            rows = session.execute(
                select(Identity.person_id, Identity.name).where(
                    Identity.person_id.in_(person_ids)
                )
            ).all()
    except Exception as exc:
        raise DatabaseError(f"Failed to resolve identity names: {exc}") from exc
    return {pid: name for pid, name in rows}


def search_embeddings(
    embedding: np.ndarray,
    limit: int = 10,
    metric: str = settings.distance_metric,
) -> List[Tuple[str, float]]:
    """Find the nearest *identities* by pgvector KNN (nearest neighbour).

    The pgvector distance operator for ``metric`` is ordered (backed by the
    HNSW index) and the nearest embedding rows are then collapsed per
    identity, returning each person once with their best similarity.

    Supported metrics: ``cosine`` (default), ``l2``, ``inner_product``.
    """
    query = embedding.flatten().astype(np.float32)
    distance, score = _distance_expr(metric, query)
    try:
        with session_scope() as session:
            rows = session.execute(
                select(
                    FaceEmbedding.person_id,
                    score.label("similarity"),
                )
                .order_by(distance)
                .limit(max(limit * 10, 100))
            ).all()
    except Exception as exc:
        raise DatabaseError(f"Failed to search embeddings: {exc}") from exc

    best: Dict[str, float] = {}
    for person_id, similarity in rows:
        if similarity > best.get(person_id, float("-inf")):
            best[person_id] = similarity
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def match_identity(
    embedding: np.ndarray,
    threshold: float = settings.recognize_threshold,
    metric: str = settings.distance_metric,
) -> Tuple[str, float]:
    """Return ``(person_id, similarity)`` of the nearest identity via pgvector.

    Falls back to ``("unknown", score)`` when the best match is below
    ``threshold``. Uses the same vector-DB nearest-neighbour search as
    ``search_embeddings``.
    """
    ranked = search_embeddings(embedding, limit=1, metric=metric)
    if not ranked:
        return "unknown", 0.0
    person_id, similarity = ranked[0]
    if similarity < threshold:
        return "unknown", similarity
    return person_id, similarity


def save_image_metadata(image: str, path: str, people: list[dict]) -> None:
    """Upsert per-image metadata: the list of people, count, and full path.

    ``people`` is a list of ``{"person_id": ..., "name": ...}`` dicts. Stored
    as a single row per image (JSONB), replacing the old row-per-person model.
    """
    try:
        with session_scope() as session:
            row = session.scalars(select(Image).where(Image.image == image)).first()
            if row is None:
                session.add(
                    Image(
                        image=image,
                        path=path,
                        face_count=len(people),
                        people=people,
                    )
                )
            else:
                row.path = path
                row.people = people
                row.face_count = len(people)
    except Exception as exc:
        raise DatabaseError(f"Failed to save image metadata for {image}: {exc}") from exc


def get_image(image: str) -> dict | None:
    """Return the aggregate metadata row for a single image."""
    try:
        with session_scope() as session:
            row = session.scalars(select(Image).where(Image.image == image)).first()
    except Exception as exc:
        raise DatabaseError(f"Failed to get image metadata for {image}: {exc}") from exc
    if row is None:
        return None
    return {
        "id": row.id,
        "image": row.image,
        "path": row.path,
        "face_count": row.face_count,
        "people": row.people,
    }


def list_person_images(person_id: str) -> list[str]:
    """Return the filenames of all images in which a person appeared."""
    try:
        with session_scope() as session:
            rows = session.execute(
                select(Image.image)
                .where(Image.people.contains([{"person_id": person_id}]))
                .order_by(Image.id.desc())
            ).all()
    except Exception as exc:
        raise DatabaseError(
            f"Failed to list images for person {person_id}: {exc}"
        ) from exc
    return [image for (image,) in rows]


def delete_identity(person_id: str) -> None:
    """Delete an identity and all of its embeddings."""
    try:
        with session_scope() as session:
            session.execute(
                delete(Identity).where(Identity.person_id == person_id)
            )
    except Exception as exc:
        raise DatabaseError(f"Failed to delete identity {person_id}: {exc}") from exc