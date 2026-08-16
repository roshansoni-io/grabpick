import uuid
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy import delete, func, select

from ..config import settings
from .connection import session_scope
from .exceptions import DatabaseError
from .models import FaceEmbedding, Identity, Image


def _distance_expr(metric: str, query: np.ndarray):
    """Return the pgvector distance expression and score formula for a metric."""
    if metric == "l2":
        distance = FaceEmbedding.vector.l2_distance(query)
        return distance, -distance
    if metric == "inner_product":
        distance = FaceEmbedding.vector.max_inner_product(query)
        return distance, -distance
    distance = FaceEmbedding.vector.cosine_distance(query)
    return distance, (1 - distance)


def save_identity(
    name: str,
    embedding: np.ndarray,
    source_image: str,
    person_id: str | None,
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


def load_database() -> Dict[str, np.ndarray]:
    """Load all embeddings grouped by identity, ready for in-memory matching."""
    try:
        with session_scope() as session:
            rows = session.execute(
                select(Identity.person_id, FaceEmbedding.vector, FaceEmbedding.id)
                .join(FaceEmbedding, isouter=True)
                .order_by(Identity.person_id, FaceEmbedding.id)
            ).all()
    except Exception as exc:
        raise DatabaseError(f"Failed to load database: {exc}") from exc

    database: Dict[str, List[np.ndarray]] = {}
    for person_id, vector, _ in rows:
        if vector is None:
            continue
        database.setdefault(person_id, []).append(np.asarray(vector, dtype=np.float32))

    return {pid: np.vstack(vectors) for pid, vectors in database.items()}


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


def search_embeddings(
    embedding: np.ndarray,
    limit: int = 10,
    metric: str = settings.distance_metric,
) -> List[Tuple[str, float]]:
    """Find nearest identities by pgvector distance for the given metric.

    Supported metrics: ``cosine`` (default), ``l2``, ``inner_product``.
    Returns ``(person_id, similarity)`` where similarity is higher for
    closer matches (>=1.0 for an exact match under cosine/L2).
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
                .limit(limit)
            ).all()
    except Exception as exc:
        raise DatabaseError(f"Failed to search embeddings: {exc}") from exc

    return [(person_id, float(similarity)) for person_id, similarity in rows]


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