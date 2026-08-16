"""Bulk-embed every face from images in a storage folder and store them in the DB.

For each detected face the embedding is matched against all identities already
in the database. If a match is found the face is skipped (it belongs to an
existing person); otherwise a new identity is created with a unique person_id
and the configured name (default: "unknown").

Usage:
    python scripts/embed_storage.py [--dir storage/originals] [--name NAME]
                                    [--rescan] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sqlalchemy import select

from app import ml
from app.config import settings
from app.database import (
    init_db,
    load_database,
    save_identity,
    save_image_metadata,
    session_scope,
)
from app.database.models import Image
from app.ml.recognition.matcher import FaceMatcher
from app.utils.logger import logger

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def is_processed(source_image: str) -> bool:
    with session_scope() as session:
        return session.scalar(
            select(Image.id).where(Image.image == source_image)
        ) is not None


def add_to_matcher(matcher: FaceMatcher, person_id: str, embedding: np.ndarray) -> None:
    """Register a newly created identity so later faces can match against it."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    if matcher.matrix is None:
        matcher.matrix = vector
        matcher.index_map = [person_id]
    else:
        matcher.matrix = np.vstack([matcher.matrix, vector])
        matcher.index_map.append(person_id)


def process_image(
    path: Path,
    matcher: FaceMatcher,
    name: str,
    rescan: bool,
    dry_run: bool,
) -> int:
    if not rescan and is_processed(path.name):
        logger.info("Skipping %s: already processed", path.name)
        return 0

    try:
        image = ml.load_image(path)
    except Exception as exc:
        logger.warning("Skipping %s: could not decode image (%s)", path.name, exc)
        return 0

    encodings = ml.face_encodings(image)
    if not encodings:
        logger.info("%s: no faces detected", path.name)
        return 0

    image_faces: list[tuple[str, str]] = []
    added = 0
    for i, encoding in enumerate(encodings, 1):
        person_id, score = matcher.match(encoding)
        if person_id != "unknown":
            logger.info(
                "Face %d from %s matches %s (%.3f); skipping",
                i, path.name, person_id, score,
            )
            image_faces.append((person_id, name))
            continue

        if dry_run:
            logger.info("[dry-run] would add new person for face %d from %s", i, path.name)
            image_faces.append(("__new__", name))
            added += 1
            continue

        new_id = save_identity(
            name=name,
            embedding=encoding,
            source_image=path.name,
        )
        add_to_matcher(matcher, new_id, encoding)
        logger.info("Added new person %s (%s) from %s", new_id, name, path.name)
        image_faces.append((new_id, name))
        added += 1

    if image_faces and not dry_run:
        people = [
            {"person_id": person_id, "name": person_name}
            for person_id, person_name in image_faces
        ]
        save_image_metadata(path.name, str(path), people)
        logger.info("Recorded %d person(s) for %s", len(people), path.name)
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=settings.originals_dir,
        help="Folder containing the images (default: storage/originals)",
    )
    parser.add_argument(
        "--name",
        default="unknown",
        help="Name assigned to newly created identities (default: unknown)",
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Reprocess images even if they already have embeddings",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and embed but do not write to the database",
    )
    args = parser.parse_args()

    if not args.dir.is_dir():
        logger.error("Directory not found: %s", args.dir)
        raise SystemExit(1)

    if not args.dry_run:
        init_db()

    matcher = FaceMatcher(
        database=load_database(), threshold=settings.recognize_threshold
    )

    images = iter_images(args.dir)
    if not images:
        logger.warning("No images found in %s", args.dir)
        raise SystemExit(0)

    logger.info("Processing %d images from %s", len(images), args.dir)
    total = 0
    for path in images:
        try:
            total += process_image(
                path, matcher, name=args.name, rescan=args.rescan, dry_run=args.dry_run
            )
        except Exception as exc:
            logger.error("Failed to process %s: %s", path.name, exc)

    logger.info("Done: %d new identities added from %d images", total, len(images))


if __name__ == "__main__":
    main()