from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..database import DatabaseError
from ..exceptions import ServiceError
from ..schemas.search import SearchResponse
from ..services.search_service import search_image_from_bytes

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search_endpoint(
    file: UploadFile = File(...),
    limit: int = Form(10, ge=1, le=100),
    recognize_threshold: float = Form(0.45, ge=0.0, le=1.0),
) -> SearchResponse:
    """Upload a query image and get ranked person matches."""
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        return search_image_from_bytes(data, limit, recognize_threshold)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc))