"""Video Input/Output handling via FFmpeg and FFprobe pipes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Generator, Optional
import numpy as np

from ecoup.cli import check_ffmpeg, _ensure_path_refreshed


@dataclass
class VideoMetadata:
    """Metadata extracted from video file."""
    width: int
    height: int
    fps: float
    fps_fraction: str
    total_frames: int
    duration_seconds: float
    video_codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    pix_fmt: str = "yuv420p"


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Extract comprehensive video metadata using ffprobe."""
    _ensure_path_refreshed()
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        raise RuntimeError("ffprobe is not installed or not in PATH.")

    video_path_str = str(Path(video_path).resolve())
    if not os.path.exists(video_path_str):
        raise FileNotFoundError(f"Video file not found: {video_path_str}")

    cmd = [
        ffprobe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path_str,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    video_stream = None
    audio_stream = None

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio" and audio_stream is None:
            audio_stream = stream

    if not video_stream:
        raise ValueError(f"No video stream found in: {video_path_str}")

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    pix_fmt = video_stream.get("pix_fmt", "yuv420p")
    video_codec = video_stream.get("codec_name", "unknown")

    # Framerate calculation
    r_frame_rate = video_stream.get("r_frame_rate", "30/1")
    try:
        frac = Fraction(r_frame_rate)
        fps = float(frac)
        fps_fraction = r_frame_rate
    except Exception:
        fps = 30.0
        fps_fraction = "30/1"

    # Total frames calculation
    nb_frames = video_stream.get("nb_frames")
    duration = 0.0

    # Try stream duration then format duration
    if "duration" in video_stream:
        try:
            duration = float(video_stream["duration"])
        except ValueError:
            pass

    if duration == 0.0 and "format" in data and "duration" in data["format"]:
        try:
            duration = float(data["format"]["duration"])
        except ValueError:
            pass

    if nb_frames and nb_frames.isdigit():
        total_frames = int(nb_frames)
    elif duration > 0 and fps > 0:
        total_frames = int(round(duration * fps))
    else:
        total_frames = 0

    has_audio = audio_stream is not None
    audio_codec = audio_stream.get("codec_name") if audio_stream else None

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        fps_fraction=fps_fraction,
        total_frames=total_frames,
        duration_seconds=duration,
        video_codec=video_codec,
        has_audio=has_audio,
        audio_codec=audio_codec,
        pix_fmt=pix_fmt,
    )


class VideoReader:
    """Reads raw RGB24 frames from video using FFmpeg pipe."""

    def __init__(self, video_path: str | Path):
        _ensure_path_refreshed()
        self.ffmpeg_bin = shutil.which("ffmpeg")
        if not self.ffmpeg_bin:
            raise RuntimeError("ffmpeg not found in PATH.")

        self.video_path = str(Path(video_path).resolve())
        self.metadata = get_video_metadata(self.video_path)
        self.frame_size = self.metadata.width * self.metadata.height * 3
        self.process: Optional[subprocess.Popen] = None
        self._current_frame_idx = 0

    def start(self) -> None:
        """Start ffmpeg decoding process."""
        if self.process is not None:
            return

        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            "-i", self.video_path,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 * 1024 * 1024,  # 10MB buffer for smooth pipeline
        )
        self._current_frame_idx = 0

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame as RGB numpy array (H, W, 3) uint8."""
        if self.process is None:
            self.start()

        assert self.process is not None and self.process.stdout is not None
        raw_bytes = self.process.stdout.read(self.frame_size)
        if len(raw_bytes) < self.frame_size:
            return None

        self._current_frame_idx += 1
        frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
            (self.metadata.height, self.metadata.width, 3)
        )
        return frame

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        if self.process is None:
            self.start()
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def close(self) -> None:
        """Terminate process and close pipe."""
        if self.process is not None:
            try:
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None

    def __enter__(self) -> VideoReader:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class VideoWriter:
    """Encodes raw RGB24 frames into video using FFmpeg with NVENC hardware acceleration."""

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float | str,
        audio_source_path: Optional[str | Path] = None,
        codec: str = "auto",
        preset: str = "p4",
        crf_or_cq: int = 21,
    ):
        _ensure_path_refreshed()
        self.ffmpeg_bin = shutil.which("ffmpeg")
        if not self.ffmpeg_bin:
            raise RuntimeError("ffmpeg not found in PATH.")

        self.output_path = str(Path(output_path).resolve())
        self.width = width
        self.height = height
        self.fps = str(fps)
        self.audio_source_path = str(Path(audio_source_path).resolve()) if audio_source_path else None
        self.codec = codec
        self.preset = preset
        self.crf_or_cq = crf_or_cq
        self.process: Optional[subprocess.Popen] = None
        self.frames_written = 0

        # Create parent output directory if needed
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def _determine_codec(self) -> str:
        """Choose between NVENC and CPU x264 based on hardware availability."""
        if self.codec != "auto":
            return self.codec

        ff_info = check_ffmpeg()
        if ff_info["h264_nvenc"]:
            return "h264_nvenc"
        return "libx264"

    def start(self) -> None:
        """Start ffmpeg encoding process."""
        if self.process is not None:
            return

        chosen_codec = self._determine_codec()

        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            "-y",  # overwrite output
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", self.fps,
            "-i", "-",  # Video input from stdin
        ]

        has_audio = False
        if self.audio_source_path and os.path.exists(self.audio_source_path):
            try:
                meta = get_video_metadata(self.audio_source_path)
                has_audio = meta.has_audio
            except Exception:
                has_audio = False

        if has_audio and self.audio_source_path:
            cmd.extend([
                "-i", self.audio_source_path,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "copy",
                "-shortest",
            ])
        else:
            cmd.extend([
                "-map", "0:v:0",
            ])

        # Codec specific encoding flags
        if chosen_codec in ("h264_nvenc", "hevc_nvenc"):
            cmd.extend([
                "-c:v", chosen_codec,
                "-preset", self.preset,
                "-cq", str(self.crf_or_cq),
                "-pix_fmt", "yuv420p",
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", str(self.crf_or_cq),
                "-pix_fmt", "yuv420p",
            ])

        cmd.append(self.output_path)

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 * 1024 * 1024,
        )
        self.frames_written = 0

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single RGB frame (H, W, 3) uint8 to encoder stdin."""
        if self.process is None:
            self.start()

        assert self.process is not None and self.process.stdin is not None
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame dimensions ({frame.shape[1]}x{frame.shape[0]}) do not match writer ({self.width}x{self.height})"
            )

        self.process.stdin.write(frame.tobytes())
        self.frames_written += 1

    def close(self) -> None:
        """Close stdin and wait for FFmpeg to finish encoding."""
        if self.process is not None:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                stderr_output = self.process.stderr.read() if self.process.stderr else b""
                self.process.wait(timeout=30)
                if self.process.returncode != 0:
                    raise RuntimeError(f"FFmpeg encoder error (code {self.process.returncode}): {stderr_output.decode('utf-8', errors='ignore')}")
            except Exception as e:
                try:
                    self.process.kill()
                except Exception:
                    pass
                raise e
            finally:
                self.process = None

    def __enter__(self) -> VideoWriter:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
