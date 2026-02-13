import hashlib
import json
import logging
from typing import cast
import redis
from app.schemas import PredictionResponse
from app.core.config import Settings

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, settings: Settings):
        self.enabled = False
        self.ttl = 86400 # 24 hours
        self.client: redis.Redis | None = None
        
        try:
            if settings.REDIS_URL:
                self.client = redis.Redis.from_url(
                    settings.REDIS_URL,
                    socket_connect_timeout=settings.REDIS_TIMEOUT
                )
                logger.info("Connecting to Redis via REDIS_URL...")
            else:
                self.client = redis.Redis(
                    host=settings.REDIS_HOST, 
                    port=settings.REDIS_PORT, 
                    db=settings.REDIS_DB, 
                    socket_connect_timeout=settings.REDIS_TIMEOUT
                )
            self.client.ping()
            self.enabled = True
            logger.info("Redis connected.")
        except redis.ConnectionError:
            logger.warning("Redis not available. Caching disabled (Dev Mode).")
        except Exception as e:
            logger.error(f"Redis error: {e}")

    def get_key(self, document_text: str) -> str:
        return hashlib.sha256(document_text.encode('utf-8')).hexdigest()

    def get(self, document_text: str) -> PredictionResponse | None:
        if not self.enabled or not self.client:
            return None
        try:
            key = self.get_key(document_text)
            data = self.client.get(key)
            if data:
                raw = cast(bytes, data)  # redis sync .get() returns bytes
                return PredictionResponse(**json.loads(raw.decode('utf-8')))
        except Exception as e:
            logger.error(f"Cache GET error: {e}")
        return None

    def set(self, document_text: str, response: PredictionResponse):
        if not self.enabled or not self.client:
            return
        try:
            key = self.get_key(document_text)
            self.client.setex(key, self.ttl, response.model_dump_json())
        except Exception as e:
            logger.error(f"Cache SET error: {e}")
