"""Shared application-level exception types for the API layer."""


class ServiceError(Exception):
    """Base error for service-layer failures, mapped to HTTP in routers."""

    status_code: int = 500

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class NotFoundError(ServiceError):
    """Raised when a requested resource does not exist."""

    status_code = 404


class ValidationError(ServiceError):
    """Raised when a request is semantically invalid."""

    status_code = 422


class ProcessingError(ServiceError):
    """Raised when image processing or recognition fails."""

    status_code = 400
