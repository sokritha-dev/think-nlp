# app/middlewares/logging.py

from logtail import LogtailHandler
import logging
import sys
from app.core.config import settings


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logger.addHandler(stream_handler)

    if settings.ENV == "production":
        handler = LogtailHandler(
            source_token=settings.BETTERSTACK_API_KEY,
            host=settings.BETTERSTACK_HOST,
        )
        logger.addHandler(handler)

    logger.info("✅ Logging initialized")
