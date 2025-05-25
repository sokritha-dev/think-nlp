# app/core/logging.py

import logging
import sys
import requests
from app.core.config import settings


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Local: log to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Production: log to BetterStack (Logtail)
    if settings.ENV == "production":

        class LogtailHandler(logging.Handler):
            def emit(self, record):
                log_entry = self.format(record)
                try:
                    requests.post(
                        "https://in.logs.betterstack.com",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {settings.BETTERSTACK_API_KEY}",
                        },
                        json={"dt": record.created, "message": log_entry},
                        timeout=3,
                    )
                except Exception:
                    pass  # Avoid crashing app on logging error

        logtail_handler = LogtailHandler()
        logtail_handler.setFormatter(formatter)
        logger.addHandler(logtail_handler)
