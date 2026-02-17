import pytest
from unittest.mock import MagicMock, AsyncMock
from app.api.main import app
from app.core.database import get_db
from app.core.security import get_api_key
from app.models import APIKey
from fastapi import HTTPException, Security, Request, Response
from fastapi.security import APIKeyHeader
from fastapi_limiter import FastAPILimiter
import redis.asyncio

# Mock Redis
redis.asyncio.from_url = MagicMock()

async def default_callback(request: Request, response: Response, pexpire: int):
    response.status_code = 429
    return {"error": "Rate limit exceeded"}

async def default_identifier(request: Request):
    return "test"

@pytest.fixture(scope="session", autouse=True)
def setup_mocks():
    # Mock Redis for Rate Limiter
    FastAPILimiter.redis = AsyncMock()
    FastAPILimiter.redis.script_load = AsyncMock(return_value="sha")
    # Make evalsha return 0 (allowed)
    FastAPILimiter.redis.evalsha = AsyncMock(return_value=0)

    # Set defaults usually set by init
    FastAPILimiter.http_callback = default_callback
    FastAPILimiter.identifier = default_identifier
    FastAPILimiter.prefix = "fastapi-limiter"
    FastAPILimiter.lua_sha = "sha"

@pytest.fixture(scope="function", autouse=True)
def override_dependencies():
    # Mock DB
    app.dependency_overrides[get_db] = lambda: MagicMock()

    # Mock Auth
    api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
    async def mock_get_api_key(api_key_header: str = Security(api_key_header_scheme)):
        if api_key_header == "dev-secret-key":
             return APIKey(key="dev-secret-key", owner="Test", is_active=True, rate_limit_per_minute=100)
        if api_key_header == "wrong-key":
             raise HTTPException(status_code=403, detail="Invalid API Key")
        if not api_key_header:
             raise HTTPException(status_code=401, detail="Missing API Key")
        return APIKey(key=api_key_header, owner="Test", is_active=True, rate_limit_per_minute=100)

    app.dependency_overrides[get_api_key] = mock_get_api_key

    yield
    app.dependency_overrides = {}
