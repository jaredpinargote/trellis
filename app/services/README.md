# `app/services/` — Business Logic Layer

Encapsulates model inference and caching behind injectable services.

## Files

| File | Purpose | Key Decisions |
|------|---------|---------------|
| `inference.py` | `ModelService` — loads the joblib model artifact, runs predict_proba, applies OOD threshold | Resolves model path relative to project root (`Path(__file__).parent.parent.parent`). Threshold can come from config OR the artifact itself (artifact takes precedence — it's model-specific). Raises `FileNotFoundError` on missing model for fail-fast. |
| `cache.py` | `CacheManager` — Redis caching with graceful degradation | Supports `REDIS_URL` (Railway) or individual host/port (local). When Redis is unavailable, `enabled=False` is set once at init — subsequent `get()`/`set()` calls return immediately with zero overhead. Uses `typing.cast(bytes, data)` to work around redis-py's Pyright stub issue. |
| `__init__.py` | Package marker | Empty. |

## Design Decisions

- **Graceful Redis degradation**: The API works identically with or without Redis. No exception propagation — all Redis errors are caught and logged. This ensures a dev can run `uvicorn app.main:app` without installing Redis.
- **`REDIS_URL` support**: Railway injects a single `REDIS_URL` env var. Locally, devs can use individual `REDIS_HOST`/`REDIS_PORT` or just let it default to `localhost:6379` (which fails gracefully).
- **Model artifact format**: `joblib.load()` returns a dict with keys `pipeline`, `threshold`, `model_version`. The pipeline is a scikit-learn `Pipeline` with a custom `DFRVectorizer` (defined in `app/transformers.py`).
