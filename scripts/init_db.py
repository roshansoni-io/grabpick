"""Initialize the database: create extension, tables, and the HNSW index."""

from __future__ import annotations

from app.database import init_db

if __name__ == "__main__":
    init_db()
    print("Database initialized: tables and HNSW index are ready.")