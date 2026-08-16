from .connection import init_db, session_scope
from .exceptions import DatabaseError
from .repositories import (
    delete_identity,
    list_identities,
    load_database,
    save_identity,
    search_embeddings,
)

__all__ = [
    "DatabaseError",
    "delete_identity",
    "init_db",
    "list_identities",
    "load_database",
    "save_identity",
    "search_embeddings",
    "session_scope",
]