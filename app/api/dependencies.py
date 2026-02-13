import threading
from fastapi import Depends
from app.core.config import Settings, get_settings
from app.services.inference import ModelService
from app.services.cache import CacheManager

# Module-level singletons (lazy-initialized, thread-safe)
_model_service: ModelService | None = None
_cache_service: CacheManager | None = None
_lock = threading.Lock()


def get_model_service(settings: Settings = Depends(get_settings)) -> ModelService:
    global _model_service
    if _model_service is None:
        with _lock:
            if _model_service is None:  # double-check after acquiring lock
                _model_service = ModelService(settings)
    return _model_service


def get_cache_service(settings: Settings = Depends(get_settings)) -> CacheManager:
    global _cache_service
    if _cache_service is None:
        with _lock:
            if _cache_service is None:  # double-check after acquiring lock
                _cache_service = CacheManager(settings)
    return _cache_service
