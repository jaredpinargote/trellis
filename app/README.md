# `app/` — Application Package

The FastAPI application, organized into three layers following separation-of-concerns.

## Structure

```
app/
├── api/          → Routes, DI, request handling
├── core/         → Cross-cutting: config, security, telemetry
├── services/     → Business logic: inference, caching
├── main.py       → Entrypoint shim
├── schemas.py    → Pydantic request/response models
└── transformers.py → Custom sklearn transformer (joblib-serialized)
```

## Files

| File | Purpose | Key Decisions |
|------|---------|---------------|
| `main.py` | Backward-compatible entrypoint — re-exports `app` from `api/main.py` | Keeps `uvicorn app.main:app` working after refactor. Thin shim only. |
| `schemas.py` | `DocumentRequest` and `PredictionResponse` Pydantic models | Field names match the case study spec exactly (`document_text`). Validation enforced via `Field(min_length=1, max_length=5000)`. |
| `transformers.py` | `DFRVectorizer` — custom TF-IDF transformer | **Must stay at `app.transformers`** — the serialized model artifact (`baseline.joblib`) references this import path. Moving it would break `joblib.load()`. |

## Architecture

```
Request → api/main.py → Depends(security, model, cache) → Response
                             ↓           ↓          ↓
                         core/       services/   services/
                       security.py  inference.py  cache.py
```
