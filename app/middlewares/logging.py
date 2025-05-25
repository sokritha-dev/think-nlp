# app/middlewares/logging.py

import logging
import sys
import requests
from app.core.config import settings


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if settings.ENV == "production":

        class LogtailHandler(logging.Handler):
            def emit(self, record):
                log_entry = self.format(record)
                try:
                    response = requests.post(
                        "https://in.logs.betterstack.com",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {settings.BETTERSTACK_API_KEY}",
                        },
                        json={
                            "dt": record.created,
                            "message": log_entry,
                        },
                        timeout=3,
                    )
                    if response.status_code != 200:
                        print(f"❌ BetterStack logging failed: {response.text}")
                except Exception as e:
                    print(f"❌ Exception while logging to BetterStack: {e}")

        logtail_handler = LogtailHandler()
        logtail_handler.setFormatter(formatter)
        logger.addHandler(logtail_handler)

    logger.info("✅ Logging system initialized")
