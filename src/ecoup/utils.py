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


def get_gpu_vram_info() -> tuple[float, float, str]:
    """Return (vram_used_gb, vram_total_gb, gpu_name) or (0.0, 0.0, '')."""
    # 1. Try pynvml (NVIDIA Management Library)
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = info.used / (1024 ** 3)
        total_gb = info.total / (1024 ** 3)
        return (used_gb, total_gb, str(name))
    except Exception:
        pass

    # 2. Try PyTorch CUDA mem_get_info
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used_gb = (total - free) / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            name = torch.cuda.get_device_name(0)
            return (used_gb, total_gb, str(name))
    except Exception:
        pass

    return (0.0, 0.0, "")

