"""Smoke tests for full end-to-end UpscalePipeline and resume mechanism."""

import os
import shutil
import subprocess
import cv2
import numpy as np
import pytest
from pathlib import Path

from ecoup.cli import _ensure_path_refreshed
from ecoup.config import AppConfig, ModelConfig, CheckpointConfig, EncodeConfig
from ecoup.pipeline import UpscalePipeline
from ecoup.video_io import get_video_metadata


@pytest.fixture
def smoke_video(tmp_path: Path) -> Path:
    """Create a 2-second test video with audio."""
    _ensure_path_refreshed()
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("FFmpeg not installed.")

    video_path = tmp_path / "smoke_input.mp4"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=2:size=160x120:rate=24",
        "-f", "lavfi",
        "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    return video_path


def mock_upscale_2x(tile: np.ndarray) -> np.ndarray:
    """Fast mock 2x upscaler using cv2."""
    h, w = tile.shape[:2]
    return cv2.resize(tile, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)


def test_pipeline_smoke_run(smoke_video: Path, tmp_path: Path):
    """End-to-end pipeline smoke test: input -> tile -> dedup -> encode -> audio."""
    out_video = tmp_path / "smoke_output.mp4"
    chk_dir = tmp_path / "checkpoints"

    cfg = AppConfig(
        model=ModelConfig(scale=2, target_scale=2),
        checkpoint=CheckpointConfig(enabled=True, dir=str(chk_dir), interval_frames=20),
        encode=EncodeConfig(codec="auto", copy_audio=True),
    )

    pipeline = UpscalePipeline(config=cfg, custom_upscale_fn=mock_upscale_2x)
    pipeline.run(input_path=smoke_video, output_path=out_video, resume=False)

    assert out_video.exists()
    assert out_video.stat().st_size > 1000

    meta_in = get_video_metadata(smoke_video)
    meta_out = get_video_metadata(out_video)

    assert meta_out.width == meta_in.width * 2
    assert meta_out.height == meta_in.height * 2
    assert meta_out.has_audio is True
    assert abs(meta_out.duration_seconds - meta_in.duration_seconds) < 0.5


def test_pipeline_resume(smoke_video: Path, tmp_path: Path):
    """Verify that an interrupted job can be resumed from checkpoints."""
    out_video = tmp_path / "resumed_output.mp4"
    chk_dir = tmp_path / "checkpoints_resume"

    cfg = AppConfig(
        model=ModelConfig(scale=2, target_scale=2),
        checkpoint=CheckpointConfig(enabled=True, dir=str(chk_dir), interval_frames=15),
        encode=EncodeConfig(codec="auto", copy_audio=True),
    )

    # 1. Run partially and cancel after 25 frames
    call_count = 0
    def interruptible_mock(tile: np.ndarray) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        if call_count >= 25:
            pipeline1.cancel()
        return mock_upscale_2x(tile)

    pipeline1 = UpscalePipeline(config=cfg, custom_upscale_fn=interruptible_mock)
    pipeline1.run(input_path=smoke_video, output_path=out_video, resume=False)

    # State file should exist with segment 1 saved
    job_dir = chk_dir / f"job_{smoke_video.stem}"
    assert (job_dir / "state.json").exists()

    # 2. Resume job
    pipeline2 = UpscalePipeline(config=cfg, custom_upscale_fn=mock_upscale_2x)
    pipeline2.run(input_path=smoke_video, output_path=out_video, resume=True)

    assert out_video.exists()
    meta_out = get_video_metadata(out_video)
    assert meta_out.width == 320
    assert meta_out.height == 240
    assert meta_out.has_audio is True


def test_pipeline_with_post_processing(smoke_video: Path, tmp_path: Path):
    """Test pipeline run with post-processing enabled (recover_detail, sharpen, film_grain)."""
    from ecoup.config import PostProcessConfig

    out_video = tmp_path / "post_processed_output.mp4"
    cfg = AppConfig(
        model=ModelConfig(scale=2, target_scale=2),
        checkpoint=CheckpointConfig(enabled=False),
        encode=EncodeConfig(codec="auto", copy_audio=True),
        post_process=PostProcessConfig(
            recover_detail=0.2,
            sharpen=0.25,
            film_grain=0.015,
        ),
    )

    pipeline = UpscalePipeline(config=cfg, custom_upscale_fn=mock_upscale_2x)
    pipeline.run(input_path=smoke_video, output_path=out_video, resume=False)

    assert out_video.exists()
    meta_out = get_video_metadata(out_video)
    assert meta_out.width == 320
    assert meta_out.height == 240
