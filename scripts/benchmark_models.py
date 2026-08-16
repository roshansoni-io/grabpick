"""Benchmark the face detector and embedder models for latency.

Generates synthetic images/faces on the fly, runs N detections and 10,000
embeddings through the ONNX models, and reports latency statistics. No data
is written to the database or disk.

Usage:
    python scripts/benchmark_models.py [--detections 100] [--embeddings 10000]
                                       [--batch-size 32] [--threads 2]
                                       [--image-size 640] [--warmup 5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.ml.detection.detector import FaceDetector
from app.ml.embedding.embedder import FaceEmbedder
from app.ml.recognition.pipeline import extract_faces
from app.ml.types import Detection


def make_image(size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a synthetic RGB image with low-noise background."""
    base = rng.integers(40, 80, (size, size, 1), dtype=np.uint8)
    noise = rng.integers(0, 25, (size, size, 3), dtype=np.uint8)
    return (base + noise).astype(np.uint8)


def make_face_crops(count: int, size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Create synthetic 112x112 face-like crops."""
    return [
        rng.integers(60, 190, (size, size, 3), dtype=np.uint8) for _ in range(count)
    ]


def report(name: str, timings: list[float]) -> None:
    arr = np.asarray(timings, dtype=np.float64)
    total = arr.sum()
    print(f"\n=== {name} ===")
    print(f"  calls        : {len(arr)}")
    print(f"  total        : {total:.3f}s")
    print(f"  mean         : {arr.mean() * 1e3:.3f} ms")
    print(f"  p50          : {np.percentile(arr, 50) * 1e3:.3f} ms")
    print(f"  p95          : {np.percentile(arr, 95) * 1e3:.3f} ms")
    print(f"  p99          : {np.percentile(arr, 99) * 1e3:.3f} ms")
    print(f"  max          : {arr.max() * 1e3:.3f} ms")
    print(f"  throughput   : {len(arr) / total:.1f} calls/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=int, default=100)
    parser.add_argument("--embeddings", type=int, default= 10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    from app.config import settings

    rng = np.random.default_rng(0)

    print("Loading models ...")
    detector = FaceDetector(
        model_path=str(settings.detector_model),
        num_threads=args.threads,
        confidence_threshold=settings.confidence_threshold,
        nms_threshold=settings.nms_threshold,
    )
    embedder = FaceEmbedder(
        model_path=str(settings.embedding_model),
        num_threads=args.threads,
    )
    input_size = (args.image_size, args.image_size)
    print(
        f"Detector  : {settings.detector_model}\n"
        f"Embedder  : {settings.embedding_model} (input {embedder.input_size})"
    )

    # --- warmup ---
    print(f"\nWarming up ({args.warmup} iterations) ...")
    for _ in range(args.warmup):
        detector.detect(make_image(args.image_size, rng), input_size)
    warm = make_face_crops(4, embedder.input_size[0], rng)
    embedder.embed(warm, batch_size=args.batch_size)

    # --- detector benchmark ---
    det_timings: list[float] = []
    faces_total = 0
    for _ in range(args.detections):
        image = make_image(args.image_size, rng)
        start = time.perf_counter()
        detections = detector.detect(image, input_size)
        det_timings.append(time.perf_counter() - start)
        faces_total += len(detections)
    report("Face Detector", det_timings)
    print(f"  total detections: {faces_total} (synthetic; detections optional)")

    # --- embedder benchmark (batched, 10,000 embeddings) ---
    total = args.embeddings
    batches = max(1, total // args.batch_size)
    remaining = total - batches * args.batch_size
    sizes = [args.batch_size] * batches + ([remaining] if remaining else [])

    emb_timings: list[float] = []
    for size in sizes:
        faces = make_face_crops(size, embedder.input_size[0], rng)
        start = time.perf_counter()
        embedder.embed(faces, batch_size=args.batch_size)
        emb_timings.append(time.perf_counter() - start)

    report("Face Embedder (batched inference)", emb_timings)
    print(
        f"  embeddings    : {sum(sizes)}\n"
        f"  batch size    : {args.batch_size}\n"
        f"  embeddings/s  : {sum(sizes) / sum(emb_timings):.1f}"
    )


if __name__ == "__main__":
    main()