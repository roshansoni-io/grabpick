from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..database import DatabaseError
from ..exceptions import ServiceError
from ..schemas.person import Person, PersonList
from ..services.person_service import (
    delete_person,
    enroll_person,
    get_person,
    list_people,
)

router = APIRouter(prefix="/api/people", tags=["people"])


@router.get("", response_model=PersonList)
def list_people_endpoint() -> PersonList:
    try:
        return PersonList(people=list_people())
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{person_id}", response_model=Person)
def get_person_endpoint(person_id: str) -> Person:
    person = get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
    return person


@router.post("", response_model=Person, status_code=201)
def enroll_person_endpoint(
    file: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=200),
) -> Person:
    """Enroll a new person from an uploaded reference image containing a face."""
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        return enroll_person(name, data, file.filename or "reference.jpg")
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{person_id}", status_code=204)
def delete_person_endpoint(person_id: str) -> None:
    try:
        deleted = delete_person(person_id)
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Person {person_id} not found")