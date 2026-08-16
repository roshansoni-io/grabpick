from typing import Dict

from fastapi import APIRouter

from .. import ml
from ..config import settings

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=Dict[str, object])
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "database": str(settings.database_url).split("@")[-1],
        "detector": str(settings.detector_model),
        "embedder": str(settings.embedding_model),
        "recognize_threshold": settings.recognize_threshold,
        "ml_version": ml.__version__,
    }
