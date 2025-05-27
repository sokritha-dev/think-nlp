from contextlib import asynccontextmanager
import logging
import time
import sqlalchemy
from fastapi import FastAPI, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi.errors import RateLimitExceeded

from app.core.database import database
from app.api import analysis, clean, eda, file, pipeline, topic_modeling, upload
from app.middlewares.access_logger import AccessLoggingMiddleware
from app.middlewares.logging import setup_logging
from app.middlewares.security import (
    SecurityHeadersMiddleware,
    add_cors_middleware,
    add_rate_limit,
)
from app.utils.exception_handlers import (
    generic_exception_handler,
    http_exception_handler,
    rate_limit_exceeded_handler,
)
from app.core.config import settings


is_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global is_ready

    max_retries = 10
    delay_seconds = 3

    for attempt in range(max_retries):
        try:
            with database._engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            logging.getLogger(__name__).info("✅ Successfully connected to Postgres!")
            is_ready = True
            break
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"❌ Postgres not ready (attempt {attempt + 1}/{max_retries}) - {e}"
            )
            time.sleep(delay_seconds)
    else:
        raise RuntimeError("🚨 Could not connect to Postgres after retries!")

    yield


# ✅ SETUP LOGGING FIRST
setup_logging()


print(f"settings::: {settings}")

app = FastAPI(
    title="NLP Pipeline API",
    description="Step-by-step NLP API for topic modeling and sentiment analysis",
    version="1.0.0",
    lifespan=lifespan,
    debug=(not settings.ENV == "production"),
)


# ===============
# Middlewares
# ===============
add_cors_middleware(app)
add_rate_limit(app)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AccessLoggingMiddleware)


# ===============
# Routers
# ===============
app.include_router(upload.router)
app.include_router(clean.router)
app.include_router(eda.router)
app.include_router(topic_modeling.router)
app.include_router(analysis.router)
app.include_router(pipeline.router)
app.include_router(file.router)


# ===============
# Health Checks
# ===============
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/liveness", status_code=204)
def liveness():
    return Response()


@app.api_route("/readiness", methods=["GET", "HEAD"], status_code=200)
def readiness():
    return {"status": "ready"} if is_ready else Response(status_code=503)


# ===============
# Global Error Handlers
# ===============
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)
