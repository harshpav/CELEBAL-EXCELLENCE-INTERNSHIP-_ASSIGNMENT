"""
Logging configuration using loguru.
Call setup_logging() once at application start.
"""

import sys

from loguru import logger

from rag_system.config import settings


def setup_logging() -> None:
    """Configure loguru with level from settings."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
        colorize=True,
    )
    logger.add(
        "logs/rag_system.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,
    )
