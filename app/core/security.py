import re
import logging
from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    HAS_PRESIDIO = True
    
    # Configure to use en_core_web_sm (small model) matching Dockerfile
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
    })
    
    analyzer: AnalyzerEngine | None = AnalyzerEngine(nlp_engine=provider.create_engine())
    logger.info("Presidio Analyzer initialized with en_core_web_sm.")
except ImportError:
    HAS_PRESIDIO = False
    analyzer = None
    logger.warning("Presidio Analyzer not found. PII detection disabled.")
except Exception as e:
    HAS_PRESIDIO = False
    analyzer = None
    logger.warning(f"Presidio Analyzer init failed: {e}")

MAX_PAYLOAD_SIZE = 2 * 1024 * 1024  # 2MB

# Patterns for injection detection
_HTML_TAG_RE = re.compile(r'<[^>]+>', re.IGNORECASE)
_SCRIPT_RE = re.compile(r'<\s*script[^>]*>.*?<\s*/\s*script\s*>', re.IGNORECASE | re.DOTALL)
_SQL_INJECTION_RE = re.compile(
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC|EXECUTE)\b\s)"
    r"|(;\s*--)"
    r"|(--\s)"
    r"|(\b(OR|AND)\b\s+\d+\s*=\s*\d+)",
    re.IGNORECASE
)
_PATH_TRAVERSAL_RE = re.compile(r'(\.\.[\\/])')
_NULL_BYTE_RE = re.compile(r'\x00')
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


async def validate_payload_size(request: Request):
    """
    Dependency to check Content-Length before body parsing.
    """
    content_length = request.headers.get('content-length')
    if content_length:
        if int(content_length) > MAX_PAYLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Payload too large. Max 2MB.")


def sanitize_text(text: str) -> str:
    """
    Sanitizes input text by removing malicious patterns.
    Returns the cleaned text. Logs warnings for detected threats.
    """
    warnings_found = []

    # 1. Detect and strip script tags
    if _SCRIPT_RE.search(text):
        warnings_found.append("SCRIPT_INJECTION")
        text = _SCRIPT_RE.sub('', text)

    # 2. Strip all remaining HTML tags
    if _HTML_TAG_RE.search(text):
        warnings_found.append("HTML_TAGS")
        text = _HTML_TAG_RE.sub('', text)

    # 3. Detect SQL injection patterns (log only, don't strip — might be legit legal text about SQL)
    if _SQL_INJECTION_RE.search(text):
        warnings_found.append("SQL_INJECTION_PATTERN")

    # 4. Detect path traversal
    if _PATH_TRAVERSAL_RE.search(text):
        warnings_found.append("PATH_TRAVERSAL")
        text = _PATH_TRAVERSAL_RE.sub('', text)

    # 5. Remove null bytes
    if _NULL_BYTE_RE.search(text):
        warnings_found.append("NULL_BYTES")
        text = _NULL_BYTE_RE.sub('', text)

    # 6. Remove control characters (keep newlines, tabs, carriage returns)
    text = _CONTROL_CHAR_RE.sub('', text)

    # 7. Normalize excessive whitespace (collapse runs of spaces)
    text = re.sub(r' {3,}', '  ', text)

    if warnings_found:
        logger.warning(f"[SECURITY] Input sanitization triggered: {warnings_found}")

    return text.strip()


def check_pii(text: str):
    """
    Scans text for PII. Raises HTTPException if PII is detected
    or if the PII detection service is unavailable (fail-closed).
    """
    if not HAS_PRESIDIO or not analyzer:
        logger.error("[SECURITY] PII detection service is unavailable. Blocking request.")
        raise HTTPException(
            status_code=500,
            detail="Security check failure: PII detection service is unavailable."
        )

    try:
        results = analyzer.analyze(
            text=text,
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN"],
            language='en'
        )
    except Exception as e:
        logger.error(f"[SECURITY] PII detection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Security check failure: PII detection analysis failed."
        )

    if results:
        pii_types = list(set([res.entity_type for res in results]))
        logger.warning(f"[SECURITY] PII Detected in request: {pii_types}")
        raise HTTPException(
            status_code=400,
            detail=f"Security Policy Violation: PII detected ({', '.join(pii_types)})."
        )
