# `tests/` — Automated Tests

Pytest test suite and Locust load testing configuration.

## Files

| File | Purpose | Coverage |
|------|---------|----------|
| `test_api.py` | API endpoint tests via FastAPI `TestClient` | 15 tests across 4 groups: Health (1), Classification (5), Validation (4), Security (5) |
| `test_security.py` | Unit tests for `sanitize_text()` function directly | 10 tests: XSS, HTML stripping, null bytes, control chars, path traversal, SQL detection, whitespace normalization |
| `locustfile.py` | Locust load test configuration | 7 traffic patterns: standard, cached, large docs, bulk random, unicode, injection, health checks |

## Running

```bash
# Unit & integration tests
pytest

# Load testing (requires running API + locust installed)
locust -f tests/locustfile.py --host http://127.0.0.1:8000
```

## Test Groups in `test_api.py`

- **TestHealth**: Verifies `/health` returns 200 with model metadata
- **TestClassification**: Tests sport, technology, politics classification + OOD detection + response schema
- **TestValidation**: Empty text, missing fields, no body, max length → all return 422
- **TestSecurity**: HTML injection, SQL injection, path traversal, null bytes, unicode → all return 200 with sanitized input

## Design Decisions

- **`TestClient` over `requests`**: Tests run without a server process — faster, simpler CI integration, deterministic.
- **Classification tests use dense, representative text**: Short texts are ambiguous by design (the model was optimized for them). Test texts are long enough to be unambiguous, testing the pipeline correctness rather than model accuracy.
- **Locust weighted tasks**: `@task(5)` for standard, `@task(3)` for cached, `@task(1)` for edge cases — simulates realistic traffic distribution.
