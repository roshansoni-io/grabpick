from fastapi import APIRouter, File, HTTPException, UploadFile

from ..exceptions import ServiceError
from ..schemas.photo import PhotoUploadResponse
from ..services.photo_service import process_upload

router = APIRouter(prefix="/api/photos", tags=["photos"])


@router.post("", response_model=PhotoUploadResponse)
def upload_photo(file: UploadFile = File(...)) -> PhotoUploadResponse:
    """Upload an image, detect faces, and return detected + matched faces."""
    try:
        data = file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}")

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        return process_upload(data, file.filename or "upload.jpg")
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")
