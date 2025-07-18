from fastapi import Request
from fastapi.middleware.base import BaseHTTPMiddleware
import time
from loguru import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests with timing"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url}")

        response = await call_next(request)

        # Log response with timing
        process_time = time.time() - start_time
        logger.info(
            f"Response: {response.status_code} "
            f"({process_time:.3f}s) {request.method} {request.url}"
        )

        return response
