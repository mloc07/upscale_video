"""Tests for VideoReader, VideoWriter, and FFmpeg video I/O."""

import os
import shutil
import subprocess
import pytest
from pathlib import Path
import numpy as np

from ecoup.cli import check_ffmpeg, _ensure_path_refreshed
from ecoup.video_io import VideoReader, VideoWriter, get_video_metadata


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """Generate a synthetic 5-second test video with audio using FFmpeg."""
    _ensure_path_refreshed()
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("FFmpeg not installed, skipping video I/O test.")

    out_dir = tmp_path_factory.mktemp("video_test")
    sample_file = out_dir / "sample_5s.mp4"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=5:size=320x240:rate=24",
        "-f", "lavfi",
        "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(sample_file),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Failed to generate test video: {res.stderr}"
    assert sample_file.exists()
    return sample_file


def test_get_video_metadata(sample_video: Path):
    meta = get_video_metadata(sample_video)
    assert meta.width == 320
    assert meta.height == 240
    assert abs(meta.fps - 24.0) < 0.1
    assert meta.has_audio is True
    assert meta.audio_codec == "aac"
    assert meta.total_frames >= 118 and meta.total_frames <= 122  # ~120 frames (5s * 24fps)


def test_video_reader(sample_video: Path):
    reader = VideoReader(sample_video)
    frames = []
    with reader:
        for frame in reader:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (240, 320, 3)
            assert frame.dtype == np.uint8
            frames.append(frame)

    assert len(frames) >= 118 and len(frames) <= 122


def test_video_passthrough_roundtrip(sample_video: Path, tmp_path: Path):
    """Passthrough test: read frames, write frames via NVENC (or fallback), check output audio and duration."""
    out_video = tmp_path / "passthrough_out.mp4"

    meta = get_video_metadata(sample_video)
    reader = VideoReader(sample_video)
    writer = VideoWriter(
        output_path=out_video,
        width=meta.width,
        height=meta.height,
        fps=meta.fps_fraction,
        audio_source_path=sample_video,
        codec="auto",
    )

    frame_count = 0
    with reader, writer:
        for frame in reader:
            writer.write_frame(frame)
            frame_count += 1

    assert frame_count >= 118
    assert out_video.exists()
    assert out_video.stat().st_size > 1000

    out_meta = get_video_metadata(out_video)
    assert out_meta.width == 320
    assert out_meta.height == 240
    assert out_meta.has_audio is True
    assert abs(out_meta.duration_seconds - meta.duration_seconds) < 0.5
