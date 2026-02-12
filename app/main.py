from fastapi import FastAPI, Depends, Request
from .schemas import DocumentRequest, PredictionResponse
from .inference import model_wrapper
from .security import validate_payload_size, check_pii
from .cache import cache

app = FastAPI(
    title="Legal Document Classifier API",
    description="Classifies legal documents into 10 categories + 'Other' (OOD detection).",
    version="1.0.0"
)

@app.post("/classify_document", response_model=PredictionResponse, dependencies=[Depends(validate_payload_size)])
async def classify_document_endpoint(request: DocumentRequest):
    """
    Classifies a legal document text.
    
    - **Payload Check**: Rejects > 2MB bodies.
    - **Cache Check**: Returns cached result if available (SHA-256 key).
    - **PII Check**: Scans for sensitive info and logs warnings.
    - **Inference**: TF-IDF + SVM with Confidence Thresholding for 'Other' class.
    """
    
    # 1. Check Cache
    # We check cache before PII check to save compute, assuming cached result implies prior successful processing
    cached = cache.get(request.text)
    if cached:
        # Add header to indicate cache hit? Not essential for response model but good for debug
        return cached

    # 2. Security Check (PII)
    # This just logs warnings, does not block
    check_pii(request.text)

    # 3. Inference
    result = model_wrapper.predict(request.text)

    # 4. Set Cache
    cache.set(request.text, result)

    return result

@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy", 
        "model_loaded": model_wrapper.pipeline is not None,
        "model_threshold": model_wrapper.threshold,
        "cache_enabled": cache.enabled
    }

if __name__ == "__main__":
    import uvicorn
    # Allow running directly for debug
    uvicorn.run(app, host="0.0.0.0", port=8000)
