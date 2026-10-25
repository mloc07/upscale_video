"""Subtitle segment data types and normalisation (merge / split / timing rules)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Optional

from .config import SegmentConfig


@dataclass
class Segment:
    start: float
    end: float
    text: str
    language: Optional[str] = None


@dataclass
class TranslatedSegment:
    start: float
    end: float
    source: str
    text: str                      # translated text (equals ``source`` if translation failed)
    language: Optional[str] = None
    failed: bool = False           # True: translation blocked/failed, original text kept
    note: str = ""


_SENTENCE_END = re.compile(r"(?<=[\.\!\?。！？…])\s*")
_CLAUSE_END = re.compile(r"(?<=[,;:、，；：])\s*")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def _is_cjk(text: str) -> bool:
    return len(_CJK.findall(text)) > max(1, len(text) // 4)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def wrap_lines(text: str, max_chars: int, max_lines: int) -> str:
    """Greedy word wrap into at most ``max_lines`` lines (CJK wraps by character)."""
    text = clean_text(text)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cjk = _is_cjk(text)
    tokens = list(text) if cjk else text.split(" ")
    sep = "" if cjk else " "
    lines: list[str] = []
    cur = ""
    for tok in tokens:
        cand = tok if not cur else cur + sep + tok
        if len(cand) <= max_chars or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = tok
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        # Re-balance: give the overflow to the last line rather than dropping text.
        head, tail = lines[: max_lines - 1], lines[max_lines - 1:]
        lines = head + [sep.join(tail)]
    return "\n".join(lines)


def _split_text(text: str, parts: int) -> list[str]:
    """Split text into ``parts`` pieces preferring sentence/clause boundaries."""
    text = clean_text(text)
    if parts <= 1:
        return [text]
    pieces = [p for p in _SENTENCE_END.split(text) if p.strip()]
    if len(pieces) < parts:
        pieces = [p for piece in pieces for p in _CLAUSE_END.split(piece) if p.strip()]
    cjk = _is_cjk(text)
    if len(pieces) < parts:
        tokens = list(text) if cjk else text.split(" ")
        sep = "" if cjk else " "
        size = max(1, -(-len(tokens) // parts))
        pieces = [sep.join(tokens[i : i + size]) for i in range(0, len(tokens), size)]
    # Greedily pack pieces into ``parts`` groups of similar length.
    total = sum(len(p) for p in pieces)
    target = total / parts
    groups: list[list[str]] = [[]]
    acc = 0
    for p in pieces:
        if groups[-1] and acc + len(p) / 2 > target and len(groups) < parts:
            groups.append([])
            acc = 0
        groups[-1].append(p)
        acc += len(p)
    sep = "" if cjk else " "
    return [sep.join(g) for g in groups if g]


def split_segment(seg: Segment, cfg: SegmentConfig) -> list[Segment]:
    """Split segments that are too long (duration or characters-per-line budget)."""
    text = clean_text(seg.text)
    dur = max(0.0, seg.end - seg.start)
    char_budget = cfg.max_chars_per_line * cfg.max_lines
    parts = 1
    if dur > cfg.max_duration:
        parts = max(parts, int(-(-dur // cfg.max_duration)))
    if len(text) > char_budget:
        parts = max(parts, int(-(-len(text) // char_budget)))
    if parts <= 1:
        return [replace(seg, text=text)]
    chunks = _split_text(text, parts)
    if len(chunks) <= 1:
        return [replace(seg, text=text)]
    total_chars = sum(len(c) for c in chunks) or 1
    out: list[Segment] = []
    t = seg.start
    for i, c in enumerate(chunks):
        share = dur * len(c) / total_chars
        if share > cfg.max_duration and len(c) < 10:
            c_dur = min(share, max(cfg.min_duration, float(len(c)) * 1.5, 3.5))
            end = t + c_dur
            out.append(Segment(start=t, end=end, text=c, language=seg.language))
            t = t + share
        else:
            end = seg.end if i == len(chunks) - 1 else t + share
            out.append(Segment(start=t, end=end, text=c, language=seg.language))
            t = end
    return out


def normalize_segments(segments: list[Segment], cfg: SegmentConfig) -> list[Segment]:
    """Sort, drop empties, merge too-short, split too-long and remove overlaps."""
    items = [replace(s, text=clean_text(s.text)) for s in segments if clean_text(s.text)]
    items.sort(key=lambda s: (s.start, s.end))
    for s in items:  # guard against inverted/zero-length timing from the ASR
        if s.end <= s.start:
            s.end = s.start + cfg.min_duration

    merged: list[Segment] = []
    for s in items:
        if merged:
            prev = merged[-1]
            gap = s.start - prev.end
            short = (prev.end - prev.start) < cfg.min_duration
            if short and gap <= cfg.merge_gap:
                cjk = _is_cjk(prev.text + s.text)
                joiner = "" if cjk else " "
                cand = prev.text + joiner + s.text
                if (s.end - prev.start) <= cfg.max_duration and len(cand) <= cfg.max_chars_per_line * cfg.max_lines:
                    prev.text = cand
                    prev.end = max(prev.end, s.end)
                    continue
        merged.append(replace(s))

    out: list[Segment] = []
    for s in merged:
        out.extend(split_segment(s, cfg))

    # Enforce a minimum duration (extend into silence when possible) and no overlaps.
    for i, s in enumerate(out):
        nxt_start = out[i + 1].start if i + 1 < len(out) else float("inf")
        if s.end - s.start < cfg.min_duration:
            s.end = min(s.start + cfg.min_duration, nxt_start)
        if s.end > nxt_start:
            s.end = nxt_start
        # Reading speed: extend (up to the next segment) if too fast.
        need = len(s.text) / cfg.max_chars_per_second
        if s.end - s.start < need:
            s.end = min(s.start + need, nxt_start)
    return [s for s in out if s.end > s.start]
