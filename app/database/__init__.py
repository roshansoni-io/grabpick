from .connection import init_db, session_scope
from .exceptions import DatabaseError
from .repositories import (
    delete_identity,
    get_image,
    list_identities,
    list_person_images,
    load_database,
    save_identity,
    save_image_metadata,
    search_embeddings,
)

__all__ = [
    "DatabaseError",
    "delete_identity",
    "get_image",
    "list_identities",
    "list_person_images",
    "load_database",
    "save_identity",
    "save_image_metadata",
    "search_embeddings",
    "session_scope",
]