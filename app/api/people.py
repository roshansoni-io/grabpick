from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import settings
from ..database import DatabaseError
from ..exceptions import ServiceError
from ..schemas.person import Person, PersonList
from ..services.person_service import (
    delete_person,
    enroll_person,
    get_person,
    list_people,
)
from ..utils.logger import logger

router = APIRouter(prefix="/api/people", tags=["people"])

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}


@router.get("", response_model=PersonList)
def list_people_endpoint() -> PersonList:
    try:
        return PersonList(people=list_people())
    except DatabaseError as exc:
        logger.error("list_people failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database operation failed.")


@router.get("/{person_id}", response_model=Person)
def get_person_endpoint(person_id: str) -> Person:
    try:
        person = get_person(person_id)
    except DatabaseError as exc:
        logger.error("get_person failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database operation failed.")

    if person is None:
        raise HTTPException(
            status_code=404,
            detail=f"Person {person_id} not found",
        )

    return person


@router.post("", response_model=Person, status_code=201)
def enroll_person_endpoint(
    file: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=200),
) -> Person:
    """Enroll a new person from an uploaded reference image containing a face."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG, and WebP images are supported",
        )

    try:
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail="Image is too large",
            )

        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        try:
            return enroll_person(name, data, file.filename or "reference.jpg")
        except ServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc))
        except DatabaseError as exc:
            logger.error("enroll_person failed: %s", exc)
            raise HTTPException(status_code=500, detail="Database operation failed.")
    finally:
        file.file.close()


@router.delete("/{person_id}", status_code=204)
def delete_person_endpoint(person_id: str) -> None:
    try:
        deleted = delete_person(person_id)
    except DatabaseError as exc:
        logger.error("delete_person failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database operation failed.")
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
