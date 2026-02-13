import time
import threading
import numpy as np
from collections import deque
from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from .schemas import DocumentRequest, PredictionResponse
from .inference import model_wrapper
from .security import validate_payload_size, sanitize_text, check_pii
from .cache import cache

app = FastAPI(
    title="Document Classifier API",
    description="Classifies documents into 10 categories + 'Other' (OOD detection).",
    version="2.0.0"
)


# ── Telemetry ──────────────────────────────────────────────────────
class Telemetry:
    """Thread-safe request telemetry collector."""
    def __init__(self, max_latencies=10000):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.latencies = deque(maxlen=max_latencies)
        self.label_counts = {}

    def record_request(self, latency_ms, label, is_cached=False, is_error=False):
        with self._lock:
            self.total_requests += 1
            if is_error:
                self.total_errors += 1
            if is_cached:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            self.latencies.append(latency_ms)
            self.label_counts[label] = self.label_counts.get(label, 0) + 1

    def snapshot(self):
        with self._lock:
            uptime = time.time() - self.start_time
            lats = np.array(self.latencies) if self.latencies else np.array([0])
            return {
                "uptime_seconds": round(uptime, 1),
                "uptime_human": _format_uptime(uptime),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": round(self.total_errors / max(self.total_requests, 1), 4),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": round(self.cache_hits / max(self.total_requests, 1), 4),
                "latency": {
                    "mean_ms": round(float(np.mean(lats)), 2),
                    "p50_ms": round(float(np.median(lats)), 2),
                    "p95_ms": round(float(np.percentile(lats, 95)), 2),
                    "p99_ms": round(float(np.percentile(lats, 99)), 2),
                    "max_ms": round(float(np.max(lats)), 2),
                    "samples": len(self.latencies),
                },
                "label_distribution": dict(sorted(
                    self.label_counts.items(), key=lambda x: -x[1]
                )),
                "requests_per_second": round(
                    self.total_requests / max(uptime, 1), 1
                ),
                "model_version": model_wrapper.model_version,
                "ood_threshold": model_wrapper.threshold,
            }


def _format_uptime(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


telemetry = Telemetry()


# ── Endpoints ──────────────────────────────────────────────────────
@app.post(
    "/classify_document",
    response_model=PredictionResponse,
    dependencies=[Depends(validate_payload_size)]
)
async def classify_document_endpoint(request: DocumentRequest):
    """
    Classifies a document text.

    Pipeline: Payload Check -> Sanitize -> Cache -> PII Scan -> Inference -> Cache Set
    """
    t0 = time.time()

    # 1. Sanitize input (strip HTML, injection patterns, control chars)
    clean_text = sanitize_text(request.document_text)

    # 2. Check Cache (on sanitized text)
    cached = cache.get(clean_text)
    if cached:
        latency = (time.time() - t0) * 1000
        telemetry.record_request(latency, cached.label, is_cached=True)
        return cached

    # 3. PII Scan (log only, does not block)
    check_pii(clean_text)

    # 4. Inference
    result = model_wrapper.predict(clean_text)

    # 5. Set Cache
    cache.set(clean_text, result)

    latency = (time.time() - t0) * 1000
    telemetry.record_request(latency, result.label)

    return result


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_wrapper.pipeline is not None,
        "model_version": model_wrapper.model_version,
        "model_threshold": model_wrapper.threshold,
        "cache_enabled": cache.enabled
    }


@app.get("/metrics")
def metrics():
    """Production telemetry: uptime, latencies, throughput, label distribution."""
    return telemetry.snapshot()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler for unhandled exceptions."""
    telemetry.record_request(0, "error", is_error=True)
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal server error: {type(exc).__name__}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
