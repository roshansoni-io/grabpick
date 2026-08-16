from dataclasses import dataclass
import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    detector_model: Path = Path(
        os.getenv("GRABPICK_DETECTOR_MODEL", "model/detector/scrfd_500m_gnkps.onnx")
    )
    embedding_model: Path = Path(
        os.getenv("GRABPICK_EMBEDDING_MODEL", "model/embedding/edgeface_xs_gamma_06.onnx")
    )
    recognize_threshold: float = float(os.getenv("GRABPICK_THRESHOLD", "0.45"))
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg://localhost:5432/grabpick"
    )
    
    providers: Tuple[str, ...] = ("CPUExecutionProvider",)
    num_threads: int = 2
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_size: Tuple[int, int] = (640, 640)
    vector_dim: int = 512
    storage_dir: Path = Path("storage")
    originals_dir: Path = Path("storage/originals")
    thumbnails_dir: Path = Path("storage/thumbnails")
    thumbnail_size: Tuple[int, int] = (320, 320)


settings = Settings()