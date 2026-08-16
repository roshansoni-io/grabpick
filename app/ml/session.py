from pathlib import Path
from typing import Optional, Sequence

import onnxruntime as ort

from .exceptions import ModelLoadError


def load_session(
    model_path: Optional[str],
    num_threads: int = 0,
    providers: Optional[Sequence[str]] = None,
) -> ort.InferenceSession:
    """Validate a model path and create an ONNX Runtime session."""
    if model_path is None:
        raise ModelLoadError("model_path is required")
    path = Path(model_path)
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")
    if path.suffix.lower() != ".onnx":
        raise ModelLoadError(f"Expected .onnx model, got: {model_path}")

    opts = ort.SessionOptions()
    if num_threads > 0:
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=providers or ["CPUExecutionProvider"],
    )