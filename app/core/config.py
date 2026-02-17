from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore .env vars not in this model
    )

    # App
    APP_NAME: str = "Trellis Document Classifier"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # Redis — supports Railway's REDIS_URL or individual fields
    REDIS_URL: str | None = None  # e.g. redis://default:pass@host:6379
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TIMEOUT: int = 1

    # Model
    MODEL_THRESHOLD: float = 0.5
    MODEL_PATH: str = "models/baseline.joblib"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/trellis"

    # Security
    API_KEY: str = "dev-secret-key"
    RATE_LIMIT_PER_MINUTE: int = 60


@lru_cache()
def get_settings() -> Settings:
    return Settings()
