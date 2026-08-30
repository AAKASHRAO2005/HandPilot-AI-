"""
utils/logger.py — Structured logging setup.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = "gesture_controller",
    level: str = "INFO",
    log_file: Optional[str] = "gesture_controller.log",
    console: bool = True,
) -> logging.Logger:
    """
    Set up and return a named logger with rotating file handler and console handler.

    Args:
        name: Logger name.
        level: Log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Path to log file. None disables file logging.
        console: Whether to also log to stdout.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if logger.handlers:
        # Avoid duplicate handlers on re-import
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file:
        fh = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# Module-level default logger
log = setup_logger()
