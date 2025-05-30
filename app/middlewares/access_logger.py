from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import logging

logger = logging.getLogger("access")


class AccessLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        x_forwarded_for = request.headers.get("x-forwarded-for")
        ip = (
            x_forwarded_for.split(",")[0].strip()
            if x_forwarded_for
            else request.client.host
        )

        path = request.url.path
        method = request.method

        logger.info(
            "🛰️ Request received", extra={"ip": ip, "path": path, "method": method}
        )

        response = await call_next(request)
        return response
