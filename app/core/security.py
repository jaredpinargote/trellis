import re
import logging
import secrets
import json
from fastapi import Request, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.models import APIKey
from app.services.cache import CacheManager
# Import locally inside dependency to avoid potential circular imports if structure changes later?
# No, let's try top level first.
from app.api.dependencies import get_cache_service

logger = logging.getLogger(__name__)

try:
    from presidio_analyzer import AnalyzerEngine # type: ignore
    from presidio_analyzer.nlp_engine import NlpEngineProvider # type: ignore
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
    Scans text for PII and logs a warning if found.
    Does NOT block the request.
    """
    if HAS_PRESIDIO and analyzer:
        results = analyzer.analyze(
            text=text,
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN"],
            language='en'
        )
        if results:
            pii_types = list(set([res.entity_type for res in results]))
            logger.warning(f"[SECURITY] PII Detected in request: {pii_types}")


api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(
    api_key_header: str = Security(api_key_header_scheme),
    db: AsyncSession = Depends(get_db),
    cache_service: CacheManager = Depends(get_cache_service)
) -> APIKey:
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )

    # 1. Check Redis Cache
    cache_key = f"apikey:{api_key_header}"
    if cache_service.enabled and cache_service.client:
        try:
            cached_data = await cache_service.client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                # Return a valid APIKey object (detached)
                return APIKey(
                    key=data['key'],
                    rate_limit_per_minute=data.get('rate_limit_per_minute', 60),
                    owner=data.get('owner', 'Cached'),
                    is_active=True
                )
        except Exception as e:
            logger.warning(f"Auth Cache Get Error: {e}")

    # 2. Check Database
    try:
        result = await db.execute(select(APIKey).where(APIKey.key == api_key_header, APIKey.is_active == True))
        api_key_obj = result.scalar_one_or_none()
    except Exception as e:
        logger.error(f"Database Auth Error: {e}")
        raise HTTPException(status_code=500, detail="Authentication Service Unavailable")

    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or Inactive API Key",
        )

    # 3. Set Cache
    if cache_service.enabled and cache_service.client:
        try:
            payload = {
                "key": api_key_obj.key,
                "rate_limit_per_minute": api_key_obj.rate_limit_per_minute,
                "owner": api_key_obj.owner
            }
            # Cache for 5 minutes
            await cache_service.client.setex(cache_key, 300, json.dumps(payload))
        except Exception as e:
            logger.warning(f"Auth Cache Set Error: {e}")

    return api_key_obj
