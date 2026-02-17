import hashlib
import json
import logging
from typing import cast
import redis.asyncio as redis
from app.schemas import PredictionResponse
from app.core.config import Settings

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, settings: Settings):
        self.enabled = False
        self.ttl = 86400 # 24 hours
        self.client: redis.Redis | None = None
        self.settings = settings
        self._init_client()

    def _init_client(self):
        try:
            if self.settings.REDIS_URL:
                self.client = redis.from_url(
                    self.settings.REDIS_URL,
                    socket_connect_timeout=self.settings.REDIS_TIMEOUT
                )
                logger.info("Connecting to Redis via REDIS_URL...")
            else:
                self.client = redis.Redis(
                    host=self.settings.REDIS_HOST,
                    port=self.settings.REDIS_PORT,
                    db=self.settings.REDIS_DB,
                    socket_connect_timeout=self.settings.REDIS_TIMEOUT
                )
            self.enabled = True
            logger.info("Redis initialized (async).")
        except Exception as e:
            logger.error(f"Redis init error: {e}")

    def get_key(self, document_text: str) -> str:
        return hashlib.sha256(document_text.encode('utf-8')).hexdigest()

    async def get(self, document_text: str) -> PredictionResponse | None:
        if not self.enabled or not self.client:
            return None
        try:
            key = self.get_key(document_text)
            data = await self.client.get(key)
            if data:
                raw = cast(bytes, data)  # async redis without decode_responses returns bytes
                return PredictionResponse(**json.loads(raw.decode('utf-8')))
        except Exception as e:
            logger.error(f"Cache GET error: {e}")
        return None

    async def set(self, document_text: str, response: PredictionResponse):
        if not self.enabled or not self.client:
            return
        try:
            key = self.get_key(document_text)
            await self.client.setex(key, self.ttl, response.model_dump_json())
        except Exception as e:
            logger.error(f"Cache SET error: {e}")
