"""GrabPick FastAPI application: face search engine API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import health, people, photos, search
from .config import settings
from .database import init_db
from .services.face_service import reload_database
from .utils.logger import logger

APP_TITLE = "GrabPick API"
VERSION = "0.1.0"
DESCRIPTION = (
    "Faces search engine for personal photos: detect, embed, and identify "
    "faces; store embeddings in PostgreSQL with pgvector."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()
    logger.info("Loading in-memory matcher...")
    reload_database()
    logger.info("GrabPick API is ready")
    yield
    logger.info("Shutting down GrabPick API")


app = FastAPI(
    title=APP_TITLE,
    version=VERSION,
    description=DESCRIPTION,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(photos.router)
app.include_router(people.router)
app.include_router(search.router)

settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
settings.originals_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(settings.thumbnails_dir)), name="static")
app.mount("/originals", StaticFiles(directory=str(settings.originals_dir)), name="originals")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)