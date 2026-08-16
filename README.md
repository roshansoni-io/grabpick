# GrabPick

Faces search engine for personal photos. Detect, embed, and identify faces;
store identity embeddings in PostgreSQL with pgvector; expose a FastAPI image API.

## Requirements

- Python 3.14+ (aarch64)
- PostgreSQL 18+ with the **pgvector** extension
- ONNX Runtime + model files (see below)

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

Models must be present at:

```
model/detector/scrfd_500m_gnkps.onnx
model/embedding/edgeface_xs_gamma_06.onnx
```

> **Note (Termux/Android):** `pip` reports platform `android-24-arm64_v8a`, so PyPI
> ships no prebuilt wheels and C/Rust packages compile from source (slow). Use the
> pure-Python stack: `fastapi==0.103.2`, `pydantic==1.10.26`, `uvicorn`, and avoid
> `uvicorn[standard]` (uvloop needs a native build).

## 2. Configure environment

```bash
cp .env.example .env
```

Adjust if needed:

```bash
DATABASE_URL=postgresql+psycopg://localhost:5432/grabpick
GRABPICK_DETECTOR_MODEL=model/detector/scrfd_500m_gnkps.onnx
GRABPICK_EMBEDDING_MODEL=model/embedding/edgeface_xs_gamma_06.onnx
GRABPICK_THRESHOLD=0.45
GRABPICK_DISTANCE_METRIC=cosine
GRABPICK_RATE_LIMIT_MAX=60
GRABPICK_RATE_LIMIT_WINDOW=60
GRABPICK_TRUST_PROXY=
```

`GRABPICK_DISTANCE_METRIC` selects the pgvector distance metric used by
`/api/search` and identity matching: `cosine` (default), `l2` (Euclidean), or
`inner_product`. Embeddings are stored as pgvector `vector(512)` columns and an
HNSW index is created per metric for fast approximate nearest-neighbour search.

Everything else has sane defaults in `app/config.py`.

## 3. Start PostgreSQL

```bash
export PGDATA=$PREFIX/var/lib/postgresql
pg_ctl -D "$PGDATA" -l "$PGDATA/logfile" -o "-p 5432 -k $PGDATA" start
```

Create the `grabpick` database if it does not exist:

```bash
psql -h 127.0.0.1 -p 5432 -d postgres -c "CREATE DATABASE grabpick;"
```

If the app connects as a non-owner role (e.g. `dbviewer`), grant it ownership
of the tables plus `public` schema access, or startup will fail with
"InsufficientPrivilege":

```bash
psql -h 127.0.0.1 -p 5432 -d grabpick \
  -c "GRANT USAGE, CREATE ON SCHEMA public TO dbviewer;"
psql -h 127.0.0.1 -p 5432 -d grabpick \
  -c "ALTER TABLE identities OWNER TO dbviewer; ALTER TABLE face_embeddings OWNER TO dbviewer; ALTER SEQUENCE face_embeddings_id_seq OWNER TO dbviewer; ALTER TABLE images OWNER TO dbviewer; ALTER SEQUENCE images_id_seq OWNER TO dbviewer;"
```

## 4. Initialize the database (once)

Creates the `vector` extension, tables, and the HNSW indexes (one per supported
distance metric):

```bash
PYTHONPATH=. python scripts/init_db.py
```

## 5. Run the API

```bash
PYTHONPATH=. python -m app
```

This starts uvicorn (with `--reload`) on `http://0.0.0.0:8000`. The app lifespan
runs `init_db()` and loads the in-memory matcher on startup.

Equivalent without the package entrypoint:

```bash
PYTHONPATH=. python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for interactive API docs.

## API endpoints

| Method | Path                     | Description                                        |
|--------|--------------------------|----------------------------------------------------|
| GET    | `/api/health`            | Service status + model/db info                     |
| GET    | `/api/people`            | List known people                                  |
| POST   | `/api/people`            | Enroll a person (multipart `file` + `name`)        |
| GET    | `/api/people/{id}`       | Get one person                                     |
| DELETE | `/api/people/{id}`       | Delete a person                                    |
| POST   | `/api/photos`            | Upload image, detect + identify faces              |
| POST   | `/api/search`            | Upload query image → ranked person matches         |
| GET    | `/api/search/{person_id}` | All images a person appeared in                     |

Static media is served at `/static` (thumbnails/face crops) and `/originals`.

### Example

```bash
# Enroll a person
curl -F "file=@face.jpg" -F "name=Alice" http://localhost:8000/api/people

# Search by image
curl -F "file=@query.jpg" http://localhost:8000/api/search

# List every image a person appeared in
curl http://localhost:8000/api/search/{person_id}

# Upload and process a photo
curl -F "file=@photo.jpg" http://localhost:8000/api/photos
```

## Batch embedding from a storage folder

`scripts/embed_storage.py` scans a folder, detects and embeds every face, and
writes them to the database. Each face is first matched against all identities
already in the DB; matches are skipped, and only new persons are added with a
unique `person_id` and the configured name (default `unknown`). It also records
per-image metadata (the people found, count, and path) in the `images` table.

```bash
PYTHONPATH=. python scripts/embed_storage.py                        # storage/originals
PYTHONPATH=. python scripts/embed_storage.py --dir /path/to/photos --name "Family"
PYTHONPATH=. python scripts/embed_storage.py --rescan --dry-run     # re-scan, no writes
```

## Model latency benchmark

`scripts/benchmark_models.py` runs the detector and embedder on synthetic data
and reports latency statistics (mean/p50/p95/p99/max, throughput). No data is
written to the database or disk.

```bash
PYTHONPATH=. python scripts/benchmark_models.py                          # 100 detections, 10k embeddings
PYTHONPATH=. python scripts/benchmark_models.py --batch-size 32 --threads 4
```

## Project layout

```
app/
├── __main__.py       `python -m app` entrypoint
├── main.py           FastAPI app (lifespan, routers, static mounts, middleware)
├── config.py         Settings (env-driven)
├── api/              health, photos, people, search routers + middleware
├── schemas/          Pydantic response/request models
├── services/         photo, face, search, person services
├── ml/               detection, embedding, recognition (ONNX)
├── database/         SQLAlchemy models, repositories, connection
└── utils/            storage, logger
scripts/              init_db.py, smoke_test.py, embed_storage.py, benchmark_models.py
model/                ONNX models (gitignored)
storage/              uploads (gitignored)
```