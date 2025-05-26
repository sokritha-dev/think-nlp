from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import logging

logger = logging.getLogger("access")


class AccessLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = request.client.host
        path = request.url.path
        method = request.method

        logger.info(
            "🛰️ Request received", extra={"ip": ip, "path": path, "method": method}
        )

        response = await call_next(request)
        return response
