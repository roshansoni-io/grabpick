class FaceRecognitionError(Exception):
    """Base class for all face recognition errors."""


class ModelLoadError(FaceRecognitionError):
    """Raised when a model fails to load."""


class DetectionError(FaceRecognitionError):
    """Raised when face detection fails."""


class EmbeddingError(FaceRecognitionError):
    """Raised when embedding extraction fails."""