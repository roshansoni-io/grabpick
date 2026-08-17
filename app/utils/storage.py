from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..config import settings
from ..exceptions import ImageDecodeError

Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

# Single source of truth for supported image formats.
# MIME type -> (PIL format name, canonical file extension).
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ("JPEG", "jpg"),
    "image/png": ("PNG", "png"),
    "image/webp": ("WEBP", "webp"),
    "image/bmp": ("BMP", "bmp"),
}
_ALLOWED_FORMATS = {fmt for fmt, _ in ALLOWED_IMAGE_TYPES.values()}
_ALLOWED_EXTENSIONS = {ext for _, ext in ALLOWED_IMAGE_TYPES.values()} | {"jpeg"}


def _ensure_dirs() -> None:
    settings.originals_dir.mkdir(parents=True, exist_ok=True)
    settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)


def content_hash(data: bytes) -> str:
    """Return a content-derived ID for an image: sha256 of its bytes.

    Using the hash as the storage filename makes identical uploads
    collide on the same path, so duplicate images are never re-written
    (dedup by content instead of caching).
    """
    return hashlib.sha256(data).hexdigest()[:32]


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB numpy array (H, W, C).

    Validates the image is a real, supported image and caps the pixel
    count to protect against decompression-bomb denial of service.
    """
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in _ALLOWED_FORMATS:
                raise ImageDecodeError(
                    f"Unsupported image format: {image.format or 'unknown'}"
                )
            width, height = image.size
            if width * height > settings.max_image_pixels:
                raise ImageDecodeError(
                    "Image dimensions exceed the maximum allowed size"
                )
            return np.asarray(image.convert("RGB"))
    except ImageDecodeError:
        raise
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise ImageDecodeError("Could not decode uploaded image") from exc


def _ext(filename: str, fallback: str = "jpg") -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix if suffix in _ALLOWED_EXTENSIONS else fallback


def safe_filename(filename: str, max_len: int = 255) -> str:
    """Sanitize a client-supplied filename: strip paths and control chars."""
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    cleaned = "".join(c for c in basename if c.isprintable() and c not in "/\\")
    cleaned = cleaned[:max_len].strip(" .")
    return cleaned or "upload"


def read_upload(file) -> bytes:
    """Read an uploaded file enforcing the max size limit.

    Returns the raw bytes; raises HTTPException(413) if the upload exceeds
    ``settings.max_upload_bytes``. The caller is responsible for closing
    the file handle.
    """
    from fastapi import HTTPException

    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return data


def save_original(data: bytes, filename: str) -> tuple[Path, bool]:
    """Persist original image bytes to storage, keyed by content hash.

    Identical images produce the same hash, so the file is written only
    once; re-uploads reuse the existing path (no duplicate storage).
    Returns ``(path, created)`` where ``created`` is True only when this
    call wrote the file (False on a dedup hit), so callers can safely
    clean up their own files without deleting shared ones.
    """
    _ensure_dirs()
    ext = _ext(filename)
    path = settings.originals_dir / f"{content_hash(data)}.{ext}"
    if path.exists():
        return path, False
    path.write_bytes(data)
    return path, True


def delete_original(path: Path) -> None:
    """Remove a stored original; silently ignore missing files."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def save_thumbnail(image: np.ndarray, size: Optional[tuple] = None) -> Path:
    """Downscale an image array, save it, and return the thumbnail path.

    The filename derives from a hash of the source pixels, so the same
    image always maps to the same thumbnail (dedup; never re-encoded).
    """
    _ensure_dirs()
    size = size or settings.thumbnail_size
    img = Image.fromarray(image.astype(np.uint8))
    img.thumbnail((size[0], size[1]), Image.LANCZOS)
    rgb = img.convert("RGB")
    data = BytesIO()
    rgb.save(data, "JPEG", quality=85)
    path = settings.thumbnails_dir / f"{content_hash(data.getvalue())}.jpg"
    if not path.exists():
        path.write_bytes(data.getvalue())
    return path


def save_face_crop(image: np.ndarray, bbox: tuple) -> Path:
    """Save a padded cropped face thumbnail and return its path."""
    _ensure_dirs()
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return save_thumbnail(image)
    return save_thumbnail(crop, (112, 112))


def thumbnail_url(path: Path) -> str:
    """Map a thumbnail storage path to a public URL under /static."""
    return f"/static/{path.name}"


def original_url(path: Path) -> str:
    """Map an original storage path to a public URL under /originals."""
    return f"/originals/{path.name}"
