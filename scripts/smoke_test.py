"""Smoke test: verify the API app builds and the DB initialises."""

from __future__ import annotations

from app.database import init_db
from app.main import app
from app.services.face_service import reload_database


def main() -> None:
    init_db()
    reload_database()
    routes = sorted({r.path for r in app.routes if r.path.startswith("/api")})
    print(f"DB initialised; matcher loaded.")
    print("Registered API routes:")
    for path in routes:
        print(f"  {path}")


if __name__ == "__main__":
    main()