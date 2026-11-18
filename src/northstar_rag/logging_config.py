"""Logging setup: call setup_logging() once when an app starts."""

import logging

from northstar_rag.config import settings


def setup_logging() -> None:
    """Send log messages to the console as: time LEVEL module: message."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
