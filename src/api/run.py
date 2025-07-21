import uvicorn
from src.api.main import app
from src.api.config import APIConfig
from src.api.middleware import RequestLoggingMiddleware
from loguru import logger

# Add middleware
app.add_middleware(RequestLoggingMiddleware)


def start_server():
    """Start the FastAPI server with production configuration"""
    config = APIConfig()

    logger.info(f"Starting Stock Alerter API on {config.host}:{config.port}")

    uvicorn.run(
        "src.api.main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        workers=config.workers,
        log_level=config.log_level,
        access_log=config.access_log,
    )


if __name__ == "__main__":
    start_server()
