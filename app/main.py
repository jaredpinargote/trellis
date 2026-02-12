from fastapi import FastAPI, Depends
from .schemas import DocumentRequest, PredictionResponse
from .inference import model_wrapper
from .security import validate_payload_size, sanitize_text, check_pii
from .cache import cache

app = FastAPI(
    title="Document Classifier API",
    description="Classifies documents into 10 categories + 'Other' (OOD detection).",
    version="1.0.0"
)


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

    # 1. Sanitize input (strip HTML, injection patterns, control chars)
    clean_text = sanitize_text(request.document_text)

    # 2. Check Cache (on sanitized text)
    cached = cache.get(clean_text)
    if cached:
        return cached

    # 3. PII Scan (log only, does not block)
    check_pii(clean_text)

    # 4. Inference
    result = model_wrapper.predict(clean_text)

    # 5. Set Cache
    cache.set(clean_text, result)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
