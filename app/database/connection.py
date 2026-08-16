from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from ..config import settings
from .models import Base

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

HNSW_OPERATOR_CLASSES = {
    "cosine": "vector_cosine_ops",
    "l2": "vector_l2_ops",
    "inner_product": "vector_ip_ops",
}


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional session; commit on success, roll back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create the vector extension, tables, and HNSW indexes.

    An HNSW index is created per supported distance metric (cosine, L2,
    inner product) so approximate nearest-neighbour search stays fast
    regardless of the configured metric.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(conn)
        for metric, ops in HNSW_OPERATOR_CLASSES.items():
            conn.execute(text(
                f"""
                CREATE INDEX IF NOT EXISTS idx_face_embeddings_vector_{metric}
                ON face_embeddings USING hnsw (vector {ops})
                """
            ))