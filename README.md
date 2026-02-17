# Trellis — Document Classification API

**Interview Deliverables:**
- 📄 [Board Report (Executive Summary)](submission_docs/BOARD_REPORT.md)
- 📝 [Self-Assessment & Reviewer Guide](submission_docs/self_assessment.md)

- 📋 [Original Case Study Requirements](submission_docs/case_study_original.md)

A production-ready REST API that classifies documents into 10 categories (business, politics, sport, etc.) with out-of-distribution (OOD) detection. Built with FastAPI, optimized via Optuna across 7 retrieval methods, and enhanced with data augmentation for short-text robustness.

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run
```bash
uvicorn app.main:app --port 8000
```
> API: `http://localhost:8000` · Docs: `http://localhost:8000/docs`

### Docker
```bash
docker-compose up -d --build
```

---

## Developer Guide

### Dependencies
- **`requirements.txt`**: Application dependencies (use for `pip install`).
- **`pyproject.toml`**: Configuration for dev tools (`pyright`, `pytest`).
- **`requirements-prod.txt`**: Optimized dependencies for the Docker build.

### Testing New Models
Training scripts (e.g., `pipelines/training/retrain_dfr.py`) now output to `models/dfr_candidate.joblib` to prevent overwriting the production baseline.

To test a candidate model:
```bash
# Run the API with the candidate model
MODEL_PATH=models/dfr_candidate.joblib uvicorn app.main:app --port 8000
```

---

## Architecture

```text
app/
├── api/                    # FastAPI endpoints & dependency injection
│   ├── main.py             # Application factory, routes, error handling
│   └── dependencies.py     # DI providers for ModelService & CacheManager
├── core/                   # Shared infrastructure
│   ├── config.py           # Centralized settings (pydantic-settings)
│   ├── security.py         # Input sanitization, PII detection
│   └── telemetry.py        # Thread-safe request metrics
├── services/               # Business logic layer
│   ├── inference.py        # ModelService — model loading & prediction
│   └── cache.py            # CacheManager — Redis caching with REDIS_URL support
├── main.py                 # Entrypoint shim (backward-compatible)
├── schemas.py              # Pydantic request/response models
└── transformers.py         # Custom DFR vectorizer (joblib-serialized)

pipelines/
├── data_prep/              # Data consolidation, splitting, validation, augmentation
└── training/               # Model training, hyperparameter search, benchmarks

scripts/                    # Operational scripts
├── demo_classify.py        # Live classification demo
├── demo_ood.py             # OOD detection demo
├── demo_stress.py          # Load testing
├── test_deployment.py      # Remote deployment verification (Railway)
├── run_full_ci_suite.py    # Pyright + Pytest + Docker build
└── verify_ecosystem.py     # Full system verification

tests/
├── test_api.py             # API endpoint tests (15 cases)
└── test_security.py        # Sanitization tests (10 cases)
```

### Design Decisions

- **Dependency Injection**: Services are injected via FastAPI's `Depends()`, replacing global singletons for testability.
- **Centralized Config**: `pydantic-settings` loads from environment variables (Railway) or `.env` (local), with sensible defaults.
- **Redis Flexibility**: Supports `REDIS_URL` (Railway's injected var) or individual `REDIS_HOST`/`REDIS_PORT`. Gracefully disables caching when Redis is unavailable.
- **joblib Compatibility**: `app/transformers.py` must remain at its import path — the serialized model references `app.transformers.DFRVectorizer`.

### Data Science Decisions

-   **DFR vs TF-IDF**: Chosen for superior Performance on long texts (94.9% F1) and tuned via Optuna.
-   **Short-Text Robustness**: We trained on **augmented data** (sentence-sliced articles) to fix the "length mismatch" between training data (long) and user queries (short).
-   **OOD Thresholding**: We use a dynamic threshold (5th percentile of validation scores) rather than a fixed 0.5 cutoff.
-   *Full analysis in [BOARD_REPORT.md](BOARD_REPORT.md).*

---

## Verification

### Automated
```bash
# Type checking (Pyright, ~5s)
pyright app

# Unit & integration tests (25 tests)
pytest

# Full CI suite (types + tests + Docker)
python scripts/run_full_ci_suite.py
```

### Live Demos (requires running server)
| Script | Tests | Expected |
|--------|-------|----------|
| `python scripts/demo_classify.py` | Accuracy | ~73% on short-text queries |
| `python scripts/demo_ood.py` | Robustness | Reject nonsense (Recall > 80%) |
| `python scripts/demo_stress.py` | Performance | p99 < 200ms under load |

### Production Deployment (Live on Railway)

> 🟢 **Live URL**: `https://trellis-production-dbbf.up.railway.app`

```bash
python scripts/test_deployment.py --url https://trellis-production-dbbf.up.railway.app
```

```text
🚀 Testing API at: https://trellis-production-dbbf.up.railway.app

1. Checking Health...
   ✅ [PASS] Status: 200 | Latency: 2061ms
      Model: optuna_dfr_v1 | Threads: 0.23252840160557384

2. Classification (Valid Input)...
   ✅ [PASS] Label: business | Conf: 0.56

3. OOD Detection...
   ✅ [PASS] Correctly identified as OOD.

✨ Deployment Verified Successfully.
```

---

## Performance

- **Accuracy**: 94.2% Test F1 (Weighted)
- **Latency**: < 200ms p99
- **Container Size**: ~350MB (multistage build)

Full details in [BOARD_REPORT.md](BOARD_REPORT.md) and [self_assessment.md](self_assessment.md).
