from pathlib import Path
from typing import Dict

from fastapi import APIRouter
from sqlalchemy.engine import make_url

from .. import ml
from ..config import settings

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=Dict[str, object])
def health() -> Dict[str, object]:
    database = make_url(settings.database_url)
    return {
        "status": "ok",
        "database": database.database,
        "detector": Path(settings.detector_model).name,
        "embedder": Path(settings.embedding_model).name,
        "recognize_threshold": settings.recognize_threshold,
        "ml_version": ml.__version__,
    }
