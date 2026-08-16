from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import settings
from ..database import DatabaseError
from ..exceptions import ServiceError
from ..schemas.search import PersonImagesResponse, SearchResponse
from ..services.search_service import search_image_from_bytes, search_images_for_person
from ..utils.logger import logger

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("/{person_id}", response_model=PersonImagesResponse)
def search_person_images(person_id: str) -> PersonImagesResponse:
    """Return all images in which a person appeared."""
    try:
        return search_images_for_person(person_id)
    except DatabaseError as exc:
        logger.error("search_person_images failed: %s", exc)
        raise HTTPException(status_code=500, detail="Search failed")


@router.post("", response_model=SearchResponse)
def search_endpoint(
    file: UploadFile = File(...),
    limit: int = Form(10, ge=1, le=100),
    recognize_threshold: float = Form(0.45, ge=0.0, le=1.0),
) -> SearchResponse:
    """Upload a query image and get ranked person matches."""
    try:
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("search_endpoint: failed to read upload: %s", exc)
        raise HTTPException(status_code=400, detail="Failed to read upload")
    finally:
        file.file.close()

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        return search_image_from_bytes(data, limit, recognize_threshold)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except DatabaseError as exc:
        logger.error("search_endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail="Search failed")