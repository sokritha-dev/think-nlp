from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, Response
import sqlalchemy
from app.core.database import database
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi.errors import RateLimitExceeded

from app.api import analysis, clean, eda, file, pipeline, topic_modeling, upload
import logging

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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


is_ready = False  # <-- New global variable


@asynccontextmanager
async def lifespan(app: FastAPI):
    global is_ready

    max_retries = 10
    delay_seconds = 3

    for attempt in range(max_retries):
        try:
            with database._engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            print("✅ Successfully connected to Postgres!")
            is_ready = True
            break
        except Exception as e:
            print(
                f"❌ Postgres not ready yet (attempt {attempt + 1}/{max_retries}) - {e}"
            )
            time.sleep(delay_seconds)
    else:
        print("🚨 Could not connect to Postgres after retries. Exiting.")
        raise RuntimeError("Database not reachable after retries!")

    yield


app = FastAPI(
    title="NLP Pipeline API",
    description="Step-by-step NLP API for topic modeling and sentiment analysis",
    version="1.0.0",
    lifespan=lifespan,
    debug=True,
)


# ===============
# Securities
# ===============
setup_logging()
add_cors_middleware(app)
add_rate_limit(app)
app.add_middleware(SecurityHeadersMiddleware)


# =================
# API Endpoints
# =================
app.include_router(upload.router)
app.include_router(clean.router)
app.include_router(eda.router)
app.include_router(topic_modeling.router)
app.include_router(analysis.router)
app.include_router(pipeline.router)
app.include_router(file.router)


# =================
# Health Checks
# =================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/liveness", status_code=204)
def liveness():
    """Simple alive check."""
    return Response()


@app.api_route("/readiness", methods=["GET", "HEAD"], status_code=200)
def readiness():
    if is_ready:
        return {"status": "ready"}
    else:
        return Response(status_code=503)


# =================
# Global Error Handler
# =================
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)
