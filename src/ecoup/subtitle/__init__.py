"""Automatic subtitle module: Whisper ASR -> Gemini translation -> SRT/ASS -> burn-in."""

from .config import SubtitleConfig  # noqa: F401
from .pipeline import SubtitlePipeline  # noqa: F401
