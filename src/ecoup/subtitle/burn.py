"""Burn (hard-sub) an ASS subtitle file into a video with FFmpeg/libass."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .audio import video_info
from .config import BurnConfig
from .srt import build_ass


def escape_filter_value(value: str) -> str:
    """Escape a path/value for use inside an FFmpeg filtergraph option (Windows + Linux safe)."""
    v = value.replace("\\", "/")
    for ch in ("\\", ":", "'", ",", "[", "]", ";", "="):
        v = v.replace(ch, "\\" + ch)
    return v


def _video_args(encode) -> list[str]:
    codec = getattr(encode, "codec", "auto") if encode is not None else "auto"
    cq = getattr(encode, "crf_or_cq", 21) if encode is not None else 21
    preset = getattr(encode, "preset", "p4") if encode is not None else "p4"
    if codec == "auto":
        codec = "libx264"
        try:
            from ..cli import check_ffmpeg
            if check_ffmpeg().get("h264_nvenc"):
                codec = "h264_nvenc"
        except Exception:
            pass
    if codec in ("h264_nvenc", "hevc_nvenc"):
        return ["-c:v", codec, "-preset", preset, "-cq", str(cq), "-pix_fmt", "yuv420p"]
    return ["-c:v", codec, "-preset", "medium", "-crf", str(cq), "-pix_fmt", "yuv420p"]


def burn_subtitles(
    video: str | Path,
    ass_path: str | Path,
    output: str | Path,
    encode=None,
    fonts_dir: Optional[str] = None,
    timeout: int = 7200,
) -> Path:
    """Render ``ass_path`` onto ``video`` copying the original audio track.

    The ASS file is copied to a temporary directory with a safe name and FFmpeg runs with that as
    working directory, so spaces / special characters in the user's path never reach the filtergraph.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH.")
    video, ass_path, output = Path(video).resolve(), Path(ass_path).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ecoup_sub_") as tmp:
        safe_ass = Path(tmp) / "subs.ass"
        shutil.copyfile(ass_path, safe_ass)
        vf = "ass=subs.ass"
        if fonts_dir:
            vf += ":fontsdir=" + escape_filter_value(str(Path(fonts_dir).resolve()))
        cmd = [
            ffmpeg, "-y", "-v", "error", "-i", str(video), "-vf", vf,
            *_video_args(encode), "-c:a", "copy", "-map", "0:v:0", "-map", "0:a?", "-movflags", "+faststart",
            str(output),
        ]
        res = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, check=False, timeout=timeout)
    if res.returncode != 0 or not output.exists():
        raise RuntimeError(f"FFmpeg subtitle burn-in failed: {res.stderr.strip()[-600:]}")
    return output


def make_ass_for_video(video: str | Path, cues, burn: BurnConfig, original_cues=None) -> str:
    info = video_info(video)
    w, h = info["width"] or 1280, info["height"] or 720
    return build_ass(cues, w, h, burn, original_cues=original_cues)


def check_libass_and_fonts(font_name: str = "Noto Sans") -> dict:
    """Environment check used by ``doctor`` / scripts: libass filters + a font with Vietnamese glyphs."""
    out = {"ffmpeg": False, "libass": False, "font_found": False, "font_name": font_name}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return out
    out["ffmpeg"] = True
    res = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, check=False)
    out["libass"] = bool(re.search(r"\bsubtitles\b", res.stdout)) and bool(re.search(r"\bass\b", res.stdout))
    fc = shutil.which("fc-list")
    if fc:
        r = subprocess.run([fc, f":family={font_name}", "family"], capture_output=True, text=True, check=False)
        out["font_found"] = bool(r.stdout.strip())
    else:  # Windows: rely on common fonts present
        out["font_found"] = any(Path(p).exists() for p in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"))
    return out
