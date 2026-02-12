from fastapi import Request, HTTPException
import logging

try:
    from presidio_analyzer import AnalyzerEngine
    HAS_PRESIDIO = True
    # Initialize once to avoid overhead
    analyzer = AnalyzerEngine() 
    print("Presidio Analyzer initialized.")
except ImportError:
    HAS_PRESIDIO = False
    analyzer = None
    print("Presidio Analyzer not found. PII detection disabled.")

MAX_PAYLOAD_SIZE = 2 * 1024 * 1024 # 2MB

async def validate_payload_size(request: Request):
    """
    Middleware-like dependency to check Content-Length.
    """
    content_length = request.headers.get('content-length')
    if content_length:
        if int(content_length) > MAX_PAYLOAD_SIZE:
             raise HTTPException(status_code=413, detail="Payload too large. Max 2MB.")
    # Note: If no content-length, we assume it's chunked or small enough for now, 
    # but production would enforce limit reading the stream.

def check_pii(text: str):
    """
    Scans text for PII and logs a warning if found.
    Does NOT block the request, per specs.
    """
    if HAS_PRESIDIO and analyzer:
        # Looking for high-sensitivity items
        results = analyzer.analyze(text=text, entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN"], language='en')
        if results:
            # We log minimal info to avoid leaking PII in logs
            pii_types = list(set([res.entity_type for res in results]))
            logging.warning(f"[SECURITY] PII Detected in request: {pii_types}")
