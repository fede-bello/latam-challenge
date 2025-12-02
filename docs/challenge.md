# LATAM Challenge: Flight Delay Prediction

## Overview

This project operationalizes a machine learning model for predicting flight delays at Santiago (SCL) airport. The implementation spans four key parts: model adaptation from a Jupyter notebook, REST API development with FastAPI, cloud deployment on Google Cloud Run, and CI/CD automation via GitHub Actions.

**Tech Stack:** Python 3.11, FastAPI, scikit-learn, Pandas, Uvicorn, Docker, Google Cloud Run, GitHub Actions.

---

## Part I: Model Implementation

The `DelayModel` class (`challenge/model.py`) encapsulates the flight delay prediction logic trained on historical SCL airport data.

**Architecture:**
- **Preprocessing:** Handles categorical features (OPERA, TIPOVUELO, MES) via one-hot encoding; selects top 10 features by importance
- **Training:** LogisticRegression with `class_weight='balanced'` to handle imbalanced delay labels
- **Persistence:** Model serialization with joblib for inference during API requests

**Key Methods:**
- `preprocess(df)`: Feature engineering and encoding
- `fit(X, y)`: Train with automatic class weighting
- `predict(X)`: Return binary predictions (0/1 for no-delay/delay)

**Development:**
Transcribed from `exploration.ipynb` with bug fixes (variable naming errors), feature selection refinement (top 10 instead of all), and test coverage validation. Model passes unit tests via `make model-test`.

---

## Part II: API Implementation

The FastAPI application (`challenge/api.py`) exposes two endpoints for model serving:

**Endpoints:**
- `GET /health` � `{"status": "OK"}` - Health check for orchestration
- `POST /predict` � Accepts `PredictRequest` with flight list, returns `PredictResponse` with predictions

**Request/Response Schemas** (Pydantic):
```
FlightData: OPERA (str), TIPOVUELO (str), MES (int)
PredictRequest: flights (list[FlightData])
PredictResponse: predict (list[int])
```

**Error Handling:**
Custom exception handler converts Pydantic validation errors to HTTP 400 responses with descriptive messages. Prediction failures return HTTP 500 with error details.

**Testing:** API passes integration tests via `make api-test` including endpoint validation and schema compliance.

---

## Part III: Cloud Deployment

**Containerization:**
- Dockerfile builds on Python 3.11-slim base
- Installs dependencies via `uv sync --frozen --no-dev`
- Exposes port 8080 via EXPOSE
- Uvicorn server runs via `python main.py` (listens on 0.0.0.0:8080)
- Health check probes `/health` endpoint at 30s intervals

**Cloud Run Setup** (us-central1):
- Service: `latam-api`
- Project: `latam-airlines-challenge`
- Service Account: `github-ci-cd@latam-airlines-challenge.iam.gserviceaccount.com`
- Configuration: 1 vCPU, 1 Gi memory, 300s timeout
- Environment variables: `GCS_MODEL_BUCKET`, `GCS_MODEL_PATH` for model artifact retrieval
- Public access enabled (`--allow-unauthenticated`)

**Service Account Permissions:**
- `roles/storage.admin` - GCR image push/pull
- `roles/iam.serviceAccountUser` - Cloud Run deployment
- `roles/artifactregistry.writer` - Artifact Registry write
- `roles/run.admin` - Cloud Run service management

---

## Part IV: CI/CD Automation

**CI Pipeline** (`ci.yml`):
- Triggers on any branch push or pull request
- Python 3.11 + uv dependency management
- Runs `make model-test` and `make api-test` with coverage reporting
- JUnit and XML coverage artifacts for CI/CD integration

**CD Pipeline** (`cd.yml`):
- Triggers on main branch push only
- GCP authentication via `google-github-actions/auth@v2` (credentials_json secret)
- Docker build and push to GCR: `gcr.io/latam-airlines-challenge/latam-api:latest`
- Cloud Run deployment with automatic service update
- Rollback on deploy failure

**Workflow Fixes Applied:**
- Migrated to v2 auth action (fixes unauthenticated request errors)
- Added explicit docker login for GCR authentication
- Fixed service account role dependencies (Storage Admin, Service Account User)

---

## Testing Strategy

| Test Type | Command | Coverage |
|-----------|---------|----------|
| Model | `make model-test` | Unit tests for preprocessing, fit, predict |
| API | `make api-test` | Endpoint validation, schema validation, error handling |
| Stress | `make stress-test` | 100 concurrent users, 60s duration via Locust |

Reports generated in `/reports` directory (HTML coverage, JUnit XML, stress metrics).

---

## Deployment Checklist

1. [X] Model implementation and testing
2. [X] FastAPI endpoints and schema validation
3. [X] Docker containerization with health checks
4. [X] Cloud Run deployment configuration
5. [X] GCP service account setup with required roles
6. [X] CI/CD pipelines with proper authentication
7. [X] API live at: `https://latam-api-[region].run.app`
