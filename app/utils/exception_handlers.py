from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi.errors import RateLimitExceeded
import traceback
import uuid


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handles expected HTTP exceptions (404, 422, etc.) with standard format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.detail["code"]
                if isinstance(exc.detail, dict)
                else "HTTP_ERROR",
                "message": exc.detail["message"]
                if isinstance(exc.detail, dict)
                else str(exc.detail),
            }
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Handles unexpected server errors (500) with unique error_id."""
    error_id = str(uuid.uuid4())[:8]  # Short UUID for error tracking

    # Print full error for backend logs
    print(f"💥 Error ID {error_id} for {request.url}\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": f"Internal server error. Reference ID: {error_id}",
            }
        },
    )


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Too Many Requests",
            "message": "You have exceeded the allowed number of requests. Please try again later.",
            "retry_after": exc.detail.get("Retry-After", None),
        },
        headers={"Retry-After": str(exc.detail.get("Retry-After", 60))},
    )
