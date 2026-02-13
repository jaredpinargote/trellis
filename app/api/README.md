# `app/api/` — API Layer

FastAPI routes, dependency injection, and request lifecycle.

## Files

| File | Purpose | Key Decisions |
|------|---------|---------------|
| `main.py` | Application factory with 3 endpoints: `/classify_document`, `/health`, `/metrics` | Pipeline: validate payload → sanitize → cache check → PII scan → inference → cache set. Global exception handler catches unhandled errors and records them in telemetry. |
| `dependencies.py` | DI providers for `ModelService` and `CacheManager` | Uses **thread-safe double-checked locking** for singletons (not `@lru_cache` — Pydantic models aren't hashable). Services are lazy-initialized on first request. |
| `__init__.py` | Package marker | Empty. |

## Design Decisions

- **`async def` endpoints**: FastAPI runs sync `Depends()` in a thread pool automatically. Endpoint is async for middleware compatibility.
- **Telemetry as module-level global**: Acceptable for a metrics collector — it's thread-safe and doesn't need DI because it has no configuration.
- **`Depends(validate_payload_size)`** on the route decorator: Runs before body parsing, rejecting oversized payloads early (2MB limit).
