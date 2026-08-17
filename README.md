GrabPick

Search your personal photo library by face. Self-hosted, private, and local.

GrabPick detects faces in your photos, generates face embeddings with ONNX Runtime, stores them in PostgreSQL using pgvector, and finds matching people using vector similarity search.

No cloud vision API is required.

---

Features

- Local face detection and embedding
- SCRFD face detection
- EdgeFace face embeddings
- PostgreSQL with pgvector
- HNSW approximate nearest-neighbour search
- Cosine, L2, and inner-product distance metrics
- Content-hash based image deduplication
- Batch photo-library indexing
- Identity-based photo search
- FastAPI REST API
- PostgreSQL as the source of truth
- No in-memory face database

---

How It Works

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

The application processes images locally. Face embeddings are stored in PostgreSQL and queried directly using pgvector.

---

Tech Stack

Component| Technology
API| FastAPI
Database| PostgreSQL
Vector Search| pgvector
Vector Index| HNSW
Face Detector| SCRFD
Face Embedder| EdgeFace-XS
ML Runtime| ONNX Runtime
ORM| SQLAlchemy
PostgreSQL Driver| Psycopg

---

Requirements

- Python 3.14+
- PostgreSQL 18+
- PostgreSQL with the pgvector extension
- ONNX Runtime
- A supported CPU architecture

Models

Place the ONNX models at:

model/
├── detector/
│   └── scrfd_500m_gnkps.onnx
└── embedding/
    └── edgeface_xs_gamma_06.onnx

On platforms such as Termux/Android, some Python packages may need to be compiled from source because prebuilt wheels are not always available.

---

Installation

Clone the repository:

git clone https://github.com/<your-username>/grabpick.git
cd grabpick

Install dependencies:

pip install -r requirements.txt

Create the environment file:

cp .env.example .env

Configure the database and other settings in ".env".

---

Configuration

GrabPick uses environment variables for configuration.

DATABASE_URL=postgresql+psycopg://localhost:5432/grabpick

GRABPICK_DISTANCE_METRIC=cosine
GRABPICK_THRESHOLD=0.45

GRABPICK_DETECTOR_MODEL=model/detector/scrfd_500m_gnkps.onnx
GRABPICK_EMBEDDING_MODEL=model/embedding/edgeface_xs_gamma_06.onnx

GRABPICK_RATE_LIMIT_MAX=60
GRABPICK_RATE_LIMIT_WINDOW=60

Configuration Options

Variable| Default| Description
"DATABASE_URL"| "postgresql+psycopg://localhost:5432/grabpick"| PostgreSQL connection URL
"GRABPICK_DISTANCE_METRIC"| "cosine"| Vector distance metric
"GRABPICK_THRESHOLD"| "0.45"| Identity matching threshold
"GRABPICK_DETECTOR_MODEL"| "model/detector/scrfd_500m_gnkps.onnx"| Detector model
"GRABPICK_EMBEDDING_MODEL"| "model/embedding/edgeface_xs_gamma_06.onnx"| Embedding model
"GRABPICK_RATE_LIMIT_MAX"| "60"| Maximum requests per window
"GRABPICK_RATE_LIMIT_WINDOW"| "60"| Rate-limit window in seconds
"GRABPICK_TRUST_PROXY"| unset| Whether to trust "X-Forwarded-For"

---

Database Setup

Create the database:

createdb grabpick

Enable pgvector:

CREATE EXTENSION IF NOT EXISTS vector;

Initialize the GrabPick schema:

PYTHONPATH=. python scripts/init_db.py

The initialization script creates the required tables, enables pgvector, and creates the HNSW indexes.

Embeddings are stored as:

vector(512)

Distance Metrics

GrabPick supports:

cosine
l2
inner_product

The selected metric is controlled by:

GRABPICK_DISTANCE_METRIC=cosine

---

Running the API

Start GrabPick with:

PYTHONPATH=. python -m app

Or run Uvicorn directly:

PYTHONPATH=. python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload

The API will be available at:

http://localhost:8000

Interactive API documentation:

http://localhost:8000/docs

---

API

Method| Endpoint| Description
"GET"| "/api/health"| Service and model status
"GET"| "/api/people"| List known people
"POST"| "/api/people"| Enroll a person
"GET"| "/api/people/{id}"| Get a person
"DELETE"| "/api/people/{id}"| Delete a person
"POST"| "/api/photos"| Upload and process a photo
"POST"| "/api/search"| Search using a face image
"GET"| "/api/search/{person_id}"| Get photos containing a person

Static thumbnails and media are served through "/static", while original images are available through "/originals".

---

Examples

Enroll a Person

curl \
  -F "file=@face.jpg" \
  -F "name=Alice" \
  http://localhost:8000/api/people

Search by Face

curl \
  -F "file=@query.jpg" \
  http://localhost:8000/api/search

Upload a Photo

curl \
  -F "file=@photo.jpg" \
  http://localhost:8000/api/photos

Get Photos for a Person

curl \
  http://localhost:8000/api/search/<person_id>

---

Batch Indexing

GrabPick can scan an existing photo library and index detected faces.

Run against the default storage directory:

PYTHONPATH=. python scripts/embed_storage.py

Scan a specific directory:

PYTHONPATH=. python scripts/embed_storage.py \
  --dir /path/to/photos

Provide a default name for newly discovered identities:

PYTHONPATH=. python scripts/embed_storage.py \
  --dir /path/to/photos \
  --name "Unknown"

Run a dry scan without modifying the database:

PYTHONPATH=. python scripts/embed_storage.py \
  --rescan \
  --dry-run

Images that have already been processed can be skipped during normal indexing.

---

Face Matching

When a query image is submitted, GrabPick performs the following process:

Query Image
    |
Face Detection
    |
Face Embedding
    |
pgvector HNSW Search
    |
Similarity Ranking
    |
Threshold Check
    |
Matching Identity

The embedding is compared directly against vectors stored in PostgreSQL.

This avoids loading the entire face database into application memory and keeps PostgreSQL as the single source of truth.

---

Image Deduplication

GrabPick uses the SHA-256 hash of an image's contents as its image identifier.

Image Bytes
    |
SHA-256
    |
Image ID

Uploading identical image data therefore does not create duplicate image records.

---

Storage

Images are stored locally:

storage/
├── originals/
└── thumbnails/

These directories should not be committed to Git.

The API exposes them through:

/originals
/static

---

Testing

Run the smoke test:

PYTHONPATH=. python scripts/smoke_test.py

Run the model benchmark:

PYTHONPATH=. python scripts/benchmark_models.py

The benchmark uses synthetic inputs and does not write to the database or modify the photo library.

---

Project Structure

grabpick/
├── app/
│   ├── __main__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   ├── health.py
│   │   ├── people.py
│   │   ├── photos.py
│   │   └── search.py
│   │
│   ├── database/
│   │   ├── models.py
│   │   ├── repositories.py
│   │   └── connection.py
│   │
│   ├── ml/
│   │   ├── detection.py
│   │   └── embedding.py
│   │
│   ├── services/
│   │   ├── face.py
│   │   ├── photo.py
│   │   ├── person.py
│   │   └── search.py
│   │
│   ├── schemas/
│   └── utils/
│
├── model/
│   ├── detector/
│   └── embedding/
│
├── scripts/
│   ├── init_db.py
│   ├── embed_storage.py
│   ├── smoke_test.py
│   └── benchmark_models.py
│
├── storage/
│   ├── originals/
│   └── thumbnails/
│
├── .env.example
├── requirements.txt
└── LICENSE

---

Privacy

GrabPick is designed for local photo libraries.

The face detection and embedding pipeline runs locally, and the application does not require a cloud face-recognition service.

Images and embeddings remain under your control.

Because face embeddings are biometric information, access to the PostgreSQL database and storage directory should be properly secured.

---

Roadmap

- Web-based photo browser
- Multi-face search
- Face-quality checks during enrollment
- Hardware acceleration
- Background library indexing
- File-system change detection
- Thumbnail caching
- Mobile-friendly frontend
- Improved identity management
- Automatic people grouping

---

License

GrabPick is distributed under the MIT License.

See ""LICENSE"" (LICENSE) for details.