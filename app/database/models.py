from datetime import datetime

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Identity(Base):
    __tablename__ = "identities"

    person_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
    )

    embeddings: Mapped[list["FaceEmbedding"]] = relationship(
        back_populates="identity", cascade="all, delete-orphan"
    )


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    person_id: Mapped[str] = mapped_column(
        ForeignKey("identities.person_id", ondelete="CASCADE"), index=True
    )
    source_image: Mapped[str] = mapped_column(String, nullable=False)
    vector: Mapped[np.ndarray] = mapped_column(Vector(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    identity: Mapped[Identity] = relationship(back_populates="embeddings")


class Image(Base):
    """Per-image aggregate metadata: the people appearing in an image.

    One row per image; ``people`` holds the list of detected people as
    ``[{"person_id": ..., "name": ...}, ...]`` plus a denormalised
    ``face_count`` so queries never need to join across person rows.
    """

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    image: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    path: Mapped[str] = mapped_column(String, nullable=False)
    face_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    people: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )