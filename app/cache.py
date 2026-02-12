import hashlib
import json
import redis
import os
from .schemas import PredictionResponse

# Config
# In docker, host might be 'redis'
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
TTL = 86400 # 24 hours

class CacheManager:
    def __init__(self):
        self.enabled = False
        try:
             # Short timeout to not block startup if Redis is down
             self.client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, socket_connect_timeout=1)
             self.client.ping()
             self.enabled = True
             print(f"Redis connected at {REDIS_HOST}:{REDIS_PORT}.")
        except redis.ConnectionError:
             print("Redis not available. Caching disabled (Dev Mode).")
        except Exception as e:
            print(f"Redis error: {e}")

    def get_key(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str):
        if not self.enabled: return None
        try:
            key = self.get_key(text)
            data = self.client.get(key)
            if data:
                return PredictionResponse(**json.loads(data))
        except Exception as e:
            print(f"Cache GET error: {e}")
        return None

    def set(self, text: str, response: PredictionResponse):
        if not self.enabled: return
        try:
            key = self.get_key(text)
            self.client.setex(key, TTL, response.json())
        except Exception as e:
            print(f"Cache SET error: {e}")

# Global instance
cache = CacheManager()
