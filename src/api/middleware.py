import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger
import json


class RequestLoggingMiddleware:
    """Simple middleware to log API requests and responses for debugging and monitoring"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        request_id = str(uuid.uuid4())[:8]  # Short UUID for readability
        start_time = time.time()

        # Log incoming request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )

        # Capture response
        response_body = b""
        status_code = 200

        async def send_wrapper(message):
            nonlocal response_body, status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)

            # Log response
            duration = time.time() - start_time
            logger.info(f"[{request_id}] {status_code} - {duration:.3f}s")

            # Log errors for debugging
            if status_code >= 400:
                try:
                    error_detail = json.loads(response_body.decode())
                    logger.warning(f"[{request_id}] Error response: {error_detail}")
                except:
                    # If response isn't JSON, log first 200 chars
                    logger.warning(
                        f"[{request_id}] Error response: {response_body[:200]}"
                    )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"[{request_id}] Request failed after {duration:.3f}s: {str(e)}"
            )
            raise
