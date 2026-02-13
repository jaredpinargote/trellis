# `app/core/` — Cross-Cutting Infrastructure

Shared utilities used across the API and service layers.

## Files

| File | Purpose | Key Decisions |
|------|---------|---------------|
| `config.py` | Centralized settings via `pydantic-settings` | Uses `SettingsConfigDict(extra="ignore")` to safely coexist with unrelated `.env` vars. Supports `REDIS_URL` (Railway) or individual `REDIS_HOST`/`REDIS_PORT` (local). `@lru_cache` on `get_settings()` ensures single instance. |
| `security.py` | Input sanitization + PII detection | Strips HTML/script tags, control chars, null bytes, path traversal. SQL patterns are **logged but not stripped** (they could be legit legal text). PII detection via Presidio is optional — gracefully disabled if not installed. |
| `telemetry.py` | Thread-safe request metrics collector | Tracks latency (p50/p95/p99), cache hit rates, label distribution, error rates. Uses `threading.Lock` for thread safety and `deque(maxlen=10000)` to cap memory. |
| `__init__.py` | Package marker | Empty. |

## Design Decisions

- **Security doesn't block**: PII detection logs warnings but never blocks requests. SQL patterns are detected for audit trails, not rejected. This avoids false positives on legal documents that contain SQL-like language.
- **Presidio is optional**: Wrapped in try/except ImportError so the API works without it (dev mode). Production includes it via `requirements.txt`.
- **Config uses `extra="ignore"`**: Prevents crashes when `.env` contains vars from other tools (e.g., Railway dashboard vars).
