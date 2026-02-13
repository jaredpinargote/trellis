import hashlib
import json
import redis
import os
from .schemas import PredictionResponse

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
TTL = 86400  # 24 hours


class CacheManager:
    def __init__(self):
        self.enabled = False
        self.client: redis.Redis | None = None
        try:
            self.client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=0, socket_connect_timeout=1
            )
            self.client.ping()
            self.enabled = True
            print(f"Redis connected at {REDIS_HOST}:{REDIS_PORT}.")
        except redis.ConnectionError:
            print("Redis not available. Caching disabled (Dev Mode).")
        except Exception as e:
            print(f"Redis error: {e}")

    def get_key(self, document_text: str) -> str:
        return hashlib.sha256(document_text.encode('utf-8')).hexdigest()

    def get(self, document_text: str):
        if not self.enabled or not self.client:
            return None
        try:
            key = self.get_key(document_text)
            data = self.client.get(key)
            if data:
                # Redis returns bytes, json.loads handles it but explicit decode is safer
                return PredictionResponse(**json.loads(data.decode('utf-8')))
        except Exception as e:
            print(f"Cache GET error: {e}")
        return None

    def set(self, document_text: str, response: PredictionResponse):
        if not self.enabled or not self.client:
            return
        try:
            key = self.get_key(document_text)
            self.client.setex(key, TTL, response.model_dump_json())
        except Exception as e:
            print(f"Cache SET error: {e}")


cache = CacheManager()
