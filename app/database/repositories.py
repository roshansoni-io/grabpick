import uuid
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy import delete, func, select

from .connection import session_scope
from .exceptions import DatabaseError
from .models import FaceEmbedding, Identity


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
) -> List[Tuple[str, float]]:
    """Find nearest identities by pgvector cosine similarity."""
    query = embedding.flatten().astype(np.float32)
    distance = FaceEmbedding.vector.cosine_distance(query)
    try:
        with session_scope() as session:
            rows = session.execute(
                select(
                    FaceEmbedding.person_id,
                    (1 - distance).label("similarity"),
                )
                .order_by(distance)
                .limit(limit)
            ).all()
    except Exception as exc:
        raise DatabaseError(f"Failed to search embeddings: {exc}") from exc

    return [(person_id, float(similarity)) for person_id, similarity in rows]


def delete_identity(person_id: str) -> None:
    """Delete an identity and all of its embeddings."""
    try:
        with session_scope() as session:
            session.execute(
                delete(Identity).where(Identity.person_id == person_id)
            )
    except Exception as exc:
        raise DatabaseError(f"Failed to delete identity {person_id}: {exc}") from exc