# GrabPick

GrabPick is a private, self-hosted face search system for your personal photo library.

It detects faces locally, generates embeddings with ONNX Runtime, stores them in PostgreSQL with `pgvector`, and finds matches with vector similarity search. No cloud vision API is required.

## What It Does

- Detects faces in local photos
- Generates face embeddings on-device
- Stores identities and vectors in PostgreSQL
- Uses `pgvector` for fast similarity search
- Deduplicates images with content hashes
- Exposes a FastAPI REST API
- Keeps PostgreSQL as the source of truth

## Architecture

```text
Photo
  |
  v
Face Detection
  |
  v
Face Embedding
  |
  v
PostgreSQL + pgvector
  |
  v
HNSW Similarity Search
  |
  v
Matching Identity
  |
  v
Matching Photos
```

All processing happens locally. Face embeddings are written to PostgreSQL and queried directly through `pgvector`.

## Stack

| Component | Technology |
| --- | --- |
| API | FastAPI |
| Database | PostgreSQL |
| Vector Search | pgvector |
| Vector Index | HNSW |
| Face Detector | SCRFD |
| Face Embedder | EdgeFace-XS |
| ML Runtime | ONNX Runtime |
| ORM | SQLAlchemy |
| PostgreSQL Driver | Psycopg |

## Requirements

- Python 3.14+
- PostgreSQL 18+
- `pgvector`
- ONNX Runtime
- A supported CPU architecture

## Models

Place the ONNX models here:

```text
model/
├── detector/
│   └── scrfd_500m_gnkps.onnx
└── embedding/
    └── edgeface_xs_gamma_06.onnx
```

On platforms such as Termux or Android, some Python packages may need to be built from source because prebuilt wheels are not always available.

## Installation

```bash
git clone https://github.com/<your-username>/grabpick.git
cd grabpick
pip install -r requirements.txt
cp .env.example .env
```

Then configure the database and runtime settings in `.env`.

## Configuration

GrabPick is configured through environment variables.

```env
DATABASE_URL=postgresql+psycopg://localhost:5432/grabpick

GRABPICK_DISTANCE_METRIC=cosine
GRABPICK_THRESHOLD=0.45

GRABPICK_DETECTOR_MODEL=model/detector/scrfd_500m_gnkps.onnx
GRABPICK_EMBEDDING_MODEL=model/embedding/edgeface_xs_gamma_06.onnx

GRABPICK_RATE_LIMIT_MAX=60
GRABPICK_RATE_LIMIT_WINDOW=60
```

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://localhost:5432/grabpick` | PostgreSQL connection URL |
| `GRABPICK_DISTANCE_METRIC` | `cosine` | Vector distance metric |
| `GRABPICK_THRESHOLD` | `0.45` | Identity matching threshold |
| `GRABPICK_DETECTOR_MODEL` | `model/detector/scrfd_500m_gnkps.onnx` | Detector model path |
| `GRABPICK_EMBEDDING_MODEL` | `model/embedding/edgeface_xs_gamma_06.onnx` | Embedding model path |
| `GRABPICK_RATE_LIMIT_MAX` | `60` | Maximum requests per window |
| `GRABPICK_RATE_LIMIT_WINDOW` | `60` | Rate-limit window in seconds |
| `GRABPICK_TRUST_PROXY` | unset | Trust `X-Forwarded-For` headers |

## Database Setup

Create the database:

```bash
createdb grabpick
```

Enable `pgvector`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Initialize the schema:

```bash
PYTHONPATH=. python scripts/init_db.py
```

The initialization script creates the required tables, enables `pgvector`, and builds the HNSW indexes.

Embeddings are stored as:

```text
vector(512)
```

## Distance Metrics

GrabPick supports:

- `cosine`
- `l2`
- `inner_product`

Set the active metric with:

```env
GRABPICK_DISTANCE_METRIC=cosine
```

## Running the API

Start the application:

```bash
PYTHONPATH=. python -m app
```

Or run Uvicorn directly:

```bash
PYTHONPATH=. python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

The API will be available at:

- `http://localhost:8000`
- `http://localhost:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service and model status |
| `GET` | `/api/people` | List known people |
| `POST` | `/api/people` | Enroll a person |
| `GET` | `/api/people/{id}` | Get a person |
| `DELETE` | `/api/people/{id}` | Delete a person |
| `POST` | `/api/photos` | Upload and process a photo |
| `POST` | `/api/search` | Search using a face image |
| `GET` | `/api/search/{person_id}` | Get photos containing a person |

Static thumbnails and media are served from `/static`, while original images are available through `/originals`.
