from fastapi import APIRouter, File, HTTPException, UploadFile

from ..config import settings
from ..exceptions import ServiceError
from ..schemas.photo import PhotoUploadResponse
from ..services.photo_service import process_upload
from ..utils.logger import logger

router = APIRouter(prefix="/api/photos", tags=["photos"])


@router.post("", response_model=PhotoUploadResponse)
def upload_photo(file: UploadFile = File(...)) -> PhotoUploadResponse:
    """Upload an image, detect faces, and return detected + matched faces."""
    try:
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("upload_photo: failed to read upload: %s", exc)
        raise HTTPException(status_code=400, detail="Failed to read upload")
    finally:
        file.file.close()

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        return process_upload(data, file.filename or "upload.jpg")
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except Exception as exc:
        logger.error("upload_photo: processing failed: %s", exc)
        raise HTTPException(status_code=500, detail="Processing failed")