import os
import sys
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from config.settings import Settings

    settings = Settings()
except ImportError:

    class Settings:
        alpha_vantage_api_key = "test_key"
        api_base_url = "https://www.alphavantage.co/query"

    settings = Settings()


def validate_environment():
    """Validate all required environment variables and configurations"""
    errors = []

    if not settings.alpha_vantage_api_key:
        errors.append("ALPHA_VANTAGE_API_KEY is required")

    required_dirs = ["logs", "data/raw", "data/processed"]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

    if settings.redis_host != "localhost" or settings.redis_port != 6379:
        try:
            import redis

            r = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                decode_responses=True,
            )
            r.ping()
            logger.info("Redis connection validated")
        except Exception as e:
            errors.append(f"Redis connection failed: {e}")

    if errors:
        for error in errors:
            logger.error(error)
        raise ValueError("Environment validation failed")

    logger.info("Environment validation passed")
