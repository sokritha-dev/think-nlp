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
        user_agent = request.headers.get("user-agent", "")

        # Skip logging known system pings (e.g. root or /health with no user-agent)
        if path in ["/", "/health"] and not user_agent:
            return await call_next(request)

        logger.info(
            "🛰️ Request received",
            extra={"ip": ip, "path": path, "method": method, "user_agent": user_agent},
        )

        response = await call_next(request)
        return response
