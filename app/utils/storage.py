from __future__ import annotations

import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..config import settings
from ..exceptions import ImageDecodeError

_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

Image.MAX_IMAGE_PIXELS = settings.max_image_pixels


def _ensure_dirs() -> None:
    settings.originals_dir.mkdir(parents=True, exist_ok=True)
    settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)


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
    return suffix if suffix in {"jpg", "jpeg", "png", "webp", "bmp"} else fallback


def safe_filename(filename: str, max_len: int = 255) -> str:
    """Sanitize a client-supplied filename: strip paths and control chars."""
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    cleaned = "".join(c for c in basename if c.isprintable() and c not in "/\\")
    cleaned = cleaned[:max_len].strip(" .")
    return cleaned or "upload"


def save_original(data: bytes, filename: str) -> Path:
    """Persist original image bytes to storage and return its path."""
    _ensure_dirs()
    ext = _ext(filename)
    path = settings.originals_dir / f"{uuid.uuid4().hex}.{ext}"
    path.write_bytes(data)
    return path


def delete_original(path: Path) -> None:
    """Remove a stored original; silently ignore missing files."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def save_thumbnail(image: np.ndarray, size: Optional[tuple] = None) -> Path:
    """Downscale an image array, save it, and return the thumbnail path."""
    _ensure_dirs()
    size = size or settings.thumbnail_size
    img = Image.fromarray(image.astype(np.uint8))
    img.thumbnail((size[0], size[1]), Image.LANCZOS)
    path = settings.thumbnails_dir / f"{uuid.uuid4().hex}.jpg"
    img.convert("RGB").save(path, "JPEG", quality=85)
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
