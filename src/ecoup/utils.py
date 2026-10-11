"""Utility functions and logger setup for Eco Video Upscaler."""

from __future__ import annotations

import logging
import time
from typing import Optional
from rich.logging import RichHandler


def setup_logger(name: str = "ecoup", verbose: bool = False) -> logging.Logger:
    """Setup a formatted logger with rich console output."""
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = RichHandler(
            level=level,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def format_duration(seconds: float) -> str:
    """Format duration in seconds to mm:ss or hh:mm:ss."""
    s = int(seconds)
    hours = s // 3600
    minutes = (s % 3600) // 60
    secs = s % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_bytes(bytes_count: int | float) -> str:
    """Format byte count into human-readable string (MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"
