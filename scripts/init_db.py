import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import select
from app.core.database import engine, Base, AsyncSessionLocal
from app.models import APIKey
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_db():
    settings = get_settings()
    logger.info(f"Connecting to database at {settings.DATABASE_URL}")

    try:
        async with engine.begin() as conn:
            # Create tables
            logger.info("Creating tables...")
            await conn.run_sync(Base.metadata.create_all)

        async with AsyncSessionLocal() as session:
            # Check if default key exists
            logger.info("Checking for default API key...")
            result = await session.execute(select(APIKey).where(APIKey.key == settings.API_KEY))
            existing_key = result.scalar_one_or_none()

            if not existing_key:
                logger.info("Creating default API key...")
                new_key = APIKey(
                    key=settings.API_KEY,
                    owner="Admin",
                    is_active=True,
                    rate_limit_per_minute=settings.RATE_LIMIT_PER_MINUTE
                )
                session.add(new_key)
                await session.commit()
                logger.info("Default API key created.")
            else:
                logger.info("Default API key already exists.")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(init_db())
