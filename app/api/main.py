import time
import logging
import redis.asyncio as redis
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from app.schemas import DocumentRequest, PredictionResponse
from app.core.security import validate_payload_size, sanitize_text, check_pii, get_api_key
from app.core.telemetry import Telemetry
from app.core.config import get_settings, Settings
from app.api.dependencies import get_model_service, get_cache_service
from app.services.inference import ModelService
from app.services.cache import CacheManager

logger = logging.getLogger(__name__)

async def get_rate_limit_key(request: Request):
    return request.headers.get("X-API-Key", "anon")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load model + cache at startup so the server is ready immediately."""
    t0 = time.time()
    settings = get_settings()

    # Initialize Rate Limiter Redis
    # We use a separate async connection for the limiter
    redis_url = settings.REDIS_URL
    if not redis_url:
        redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"

    try:
        redis_conn = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await FastAPILimiter.init(redis_conn)
        logger.info("Rate Limiter initialized.")
    except Exception as e:
        logger.warning(f"Rate Limiter init failed: {e}")
        redis_conn = None

    model = get_model_service(settings)
    cache = get_cache_service(settings)
    elapsed = time.time() - t0
    logger.info(f"🟢 Ready to serve — model={model.model_version}, cache={'on' if cache.enabled else 'off'}, loaded in {elapsed:.1f}s")

    yield

    if redis_conn:
        await redis_conn.close()

app = FastAPI(
    title="Document Classifier API",
    description="Classifies documents into 10 categories + 'Other' (OOD detection).",
    version="2.0.0",
    lifespan=lifespan,
)

# Telemetry is singleton-ish by nature of being a collector
# We can inject it too, but global is acceptable for telemetry if it's thread-safe
telemetry = Telemetry()

@app.post(
    "/classify_document",
    response_model=PredictionResponse,
    dependencies=[
        Depends(validate_payload_size),
        Depends(get_api_key),
        Depends(RateLimiter(times=get_settings().RATE_LIMIT_PER_MINUTE, seconds=60, identifier=get_rate_limit_key))
    ]
)
async def classify_document_endpoint(
    request: DocumentRequest,
    model_service: ModelService = Depends(get_model_service),
    cache_service: CacheManager = Depends(get_cache_service)
):
    """
    Classifies a document text.

    Pipeline: Payload Check -> Sanitize -> Cache -> PII Scan -> Inference -> Cache Set
    """
    t0 = time.time()

    # 1. Sanitize input (strip HTML, injection patterns, control chars)
    # CPU-bound: run in threadpool
    clean_text = await run_in_threadpool(sanitize_text, request.document_text)

    # 2. Check Cache
    cached = await cache_service.get(clean_text)
    if cached:
        latency = (time.time() - t0) * 1000
        telemetry.record_request(latency, cached.label, is_cached=True)
        return cached

    # 3. PII Scan (log only, does not block)
    # CPU-bound: run in threadpool
    await run_in_threadpool(check_pii, clean_text)

    # 4. Inference
    # CPU-bound: run in threadpool (via model_service wrapper)
    result = await model_service.predict_async(clean_text)

    # 5. Set Cache
    await cache_service.set(clean_text, result)

    latency = (time.time() - t0) * 1000
    telemetry.record_request(latency, result.label)

    return result


@app.get("/health")
def health(
    model_service: ModelService = Depends(get_model_service),
    cache_service: CacheManager = Depends(get_cache_service)
):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_service.pipeline is not None,
        "model_version": model_service.model_version,
        "model_threshold": model_service.threshold,
        "cache_enabled": cache_service.enabled
    }


@app.get("/metrics")
def metrics(
    model_service: ModelService = Depends(get_model_service)
):
    """Production telemetry: uptime, latencies, throughput, label distribution."""
    # Pass metadata to telemetry snapshot
    return telemetry.snapshot(
        model_version=model_service.model_version,
        ood_threshold=model_service.threshold
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler for unhandled exceptions."""
    telemetry.record_request(0, "error", is_error=True)
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal server error: {type(exc).__name__}"}
    )

# Removed __main__ block as this should be run via uvicorn CLI
