"""SRT read/write and ASS generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

from .config import BurnConfig
from .segments import Segment, TranslatedSegment, wrap_lines


def format_ts(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


def parse_ts(text: str) -> float:
    m = _TS.fullmatch(text.strip())
    if not m:
        raise ValueError(f"Invalid SRT timestamp: {text!r}")
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def validate_timeline(items: Sequence[tuple[float, float]]) -> None:
    prev_end = 0.0
    for i, (start, end) in enumerate(items):
        if end <= start:
            raise ValueError(f"Segment {i + 1}: end ({end}) must be after start ({start}).")
        if start + 1e-6 < prev_end:
            raise ValueError(f"Segment {i + 1}: overlaps previous (start {start} < prev end {prev_end}).")
        prev_end = end


def write_srt(path: str | Path, cues: Iterable[tuple[float, float, str]], validate: bool = True) -> Path:
    cues = list(cues)
    if validate:
        validate_timeline([(a, b) for a, b, _ in cues])
    lines: list[str] = []
    for i, (start, end, text) in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{format_ts(start)} --> {format_ts(end)}")
        lines.append(text.strip() or " ")
        lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def read_srt(path: str | Path) -> list[tuple[float, float, str]]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\s*\r?\n", raw.strip())
    out: list[tuple[float, float, str]] = []
    for block in blocks:
        rows = [r for r in block.splitlines() if r.strip() != ""]
        if len(rows) < 2:
            continue
        idx = 1 if "-->" in rows[1] else 0
        if "-->" not in rows[idx]:
            continue
        a, b = (x.strip() for x in rows[idx].split("-->"))
        out.append((parse_ts(a), parse_ts(b.split()[0]), "\n".join(rows[idx + 1:])))
    return out


def segments_to_cues(segs: Sequence[Segment], max_chars: int, max_lines: int) -> list[tuple[float, float, str]]:
    return [(s.start, s.end, wrap_lines(s.text, max_chars, max_lines)) for s in segs]


def translated_to_cues(segs: Sequence[TranslatedSegment], max_chars: int, max_lines: int) -> list[tuple[float, float, str]]:
    return [(s.start, s.end, wrap_lines(s.text, max_chars, max_lines)) for s in segs]


def _ass_ts(seconds: float) -> str:
    cs_total = max(0, int(round(seconds * 100)))
    h, rem = divmod(cs_total, 360_000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    text = text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
    return text.replace("\r", "").replace("\n", "\\N")


def build_ass(
    cues: Sequence[tuple[float, float, str]],
    width: int,
    height: int,
    burn: BurnConfig,
    original_cues: Sequence[tuple[float, float, str]] | None = None,
) -> str:
    """Build an ASS document whose style scales with the video resolution (9:16 safe)."""
    font_size = max(12, int(round(height * burn.font_scale)))
    outline = max(1, int(round(font_size * burn.outline_scale)))
    margin_v = int(round(height * burn.margin_v_scale))
    margin_h = int(round(width * burn.margin_h_scale))
    bold = -1 if burn.bold else 0
    small = max(10, int(round(font_size * 0.7)))
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{burn.font_name},{font_size},{burn.primary_colour},&H000000FF,{burn.outline_colour},"
        f"&H64000000,{bold},0,0,0,100,100,0,0,1,{outline},1,2,{margin_h},{margin_h},{margin_v},1\n"
        f"Style: Orig,{burn.font_name},{small},&H00D0D0D0,&H000000FF,{burn.outline_colour},"
        f"&H64000000,0,0,0,0,100,100,0,0,1,{max(1, outline - 1)},1,2,{margin_h},{margin_h},{margin_v + font_size * 2 + 6},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    rows = [
        f"Dialogue: 0,{_ass_ts(a)},{_ass_ts(b)},Default,,0,0,0,,{_ass_escape(t)}"
        for a, b, t in cues
    ]
    if original_cues:
        rows += [
            f"Dialogue: 0,{_ass_ts(a)},{_ass_ts(b)},Orig,,0,0,0,,{_ass_escape(t)}"
            for a, b, t in original_cues
        ]
    return header + "\n".join(rows) + "\n"
