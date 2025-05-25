# app/middlewares/security.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from app.core.config import settings


# ----------------------------
# Rate Limiting Configuration
# ----------------------------
limiter = Limiter(key_func=get_remote_address)
limiter.default_limits = ["10/hour"]  # Customize as needed


def add_rate_limit(app: FastAPI) -> None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ----------------------------
# CORS Middleware
# ----------------------------
def add_cors_middleware(app: FastAPI) -> None:
    allow_origins = ["*"] if settings.ENV == "local" else [settings.FRONTEND_ORIGIN]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ----------------------------
# Security Headers Middleware
# ----------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # Enforce HTTPS
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains"
        )

        # MIME sniffing protection
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent Clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Block inline script injection (CSP)
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        # Don't leak referrer info
        response.headers["Referrer-Policy"] = "no-referrer"

        # Restrict device APIs
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"

        # Prevent third-party sites from loading our assets
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"

        # Enforce browser isolation for cross-origin embeds
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"

        return response
