"""Translator backend interface."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

from ..segments import Segment, TranslatedSegment


class TranslatorBackend(Protocol):
    def translate(
        self,
        batch: Sequence[Segment],
        context_before: Sequence[Segment] = (),
        context_after: Sequence[Segment] = (),
        source_language: Optional[str] = None,
    ) -> list[TranslatedSegment]:
        """Translate ``batch``; output length MUST equal input length and keep order."""
        ...
