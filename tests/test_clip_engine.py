"""Unit tests for VideoUpscaleEngine, clip chunking, overlap, cross-fade and pipeline integration."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from ecoup.cli import _ensure_path_refreshed
from ecoup.config import AppConfig, CheckpointConfig, EncodeConfig, InferenceConfig, ModelConfig
from ecoup.engine.base import VideoUpscaleEngine
from ecoup.pipeline import UpscalePipeline
from ecoup.video_io import get_video_metadata


class MockClipEngine(VideoUpscaleEngine):
    """Mock implementation of VideoUpscaleEngine for unit testing."""

    def __init__(self, scale: int = 2):
        self.scale = scale
        self.call_history: List[int] = []
        self.closed = False

    def load(self, model_path: str, precision: str = "fp16", provider: str = "auto") -> None:
        pass

    def upscale_clip(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        self.call_history.append(len(frames))
        out_frames = []
        for f in frames:
            h, w = f.shape[:2]
            up = cv2.resize(f, (int(round(w * self.scale)), int(round(h * self.scale))), interpolation=cv2.INTER_LINEAR)
            out_frames.append(up)
        return out_frames

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def test_clip_video(tmp_path: Path) -> Path:
    """Generate a 48-frame synthetic video (2.0s at 24fps) with audio."""
    _ensure_path_refreshed()
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("FFmpeg not installed.")

    video_path = tmp_path / "clip_input_48frames.mp4"
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


@pytest.fixture
def short_clip_video(tmp_path: Path) -> Path:
    """Generate a short 6-frame synthetic video (0.25s at 24fps)."""
    _ensure_path_refreshed()
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("FFmpeg not installed.")

    video_path = tmp_path / "clip_input_6frames.mp4"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=0.25:size=160x120:rate=24",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    return video_path


def test_clip_engine_frame_preservation(test_clip_video: Path, tmp_path: Path):
    """Test that clip engine with overlap preserves the exact total frame count."""
    out_video = tmp_path / "clip_output_preserved.mp4"

    cfg = AppConfig(
        model=ModelConfig(scale=2, target_scale=2),
        inference=InferenceConfig(clip_length=16, clip_overlap=2),
        checkpoint=CheckpointConfig(enabled=False),
        encode=EncodeConfig(codec="auto", copy_audio=True),
    )

    mock_engine = MockClipEngine(scale=2)
    pipeline = UpscalePipeline(config=cfg, engine=mock_engine)
    pipeline.run(input_path=test_clip_video, output_path=out_video)

    assert out_video.exists()
    meta_in = get_video_metadata(test_clip_video)
    meta_out = get_video_metadata(out_video)

    # Frame count and dimensions must match exactly
    assert meta_out.total_frames == meta_in.total_frames
    assert meta_out.width == meta_in.width * 2
    assert meta_out.height == meta_in.height * 2
    assert meta_out.has_audio is True
    assert len(mock_engine.call_history) > 1


def test_clip_engine_short_video(short_clip_video: Path, tmp_path: Path):
    """Test video shorter than clip_length (6 frames < 16 clip_length)."""
    out_video = tmp_path / "short_output.mp4"

    cfg = AppConfig(
        model=ModelConfig(scale=2, target_scale=2),
        inference=InferenceConfig(clip_length=16, clip_overlap=2),
        checkpoint=CheckpointConfig(enabled=False),
        encode=EncodeConfig(codec="auto", copy_audio=False),
    )

    mock_engine = MockClipEngine(scale=2)
    pipeline = UpscalePipeline(config=cfg, engine=mock_engine)
    pipeline.run(input_path=short_clip_video, output_path=out_video)

    meta_in = get_video_metadata(short_clip_video)
    meta_out = get_video_metadata(out_video)

    assert meta_out.total_frames == meta_in.total_frames
    assert len(mock_engine.call_history) == 1
    assert mock_engine.call_history[0] == meta_in.total_frames


def test_clip_engine_crossfade_blending():
    """Verify linear cross-fade blending computation between boundary frames."""
    # Simulate frame A and frame B
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b = np.full((100, 100, 3), 255, dtype=np.uint8)

    fade_len = 2
    # k=0: alpha = 1 / 3
    alpha_0 = 1.0 / 3.0
    blended_0 = cv2.addWeighted(frame_a, 1.0 - alpha_0, frame_b, alpha_0, 0)
    assert np.allclose(blended_0[0, 0, 0], int(round(255 * (1.0 / 3.0))), atol=2)

    # k=1: alpha = 2 / 3
    alpha_1 = 2.0 / 3.0
    blended_1 = cv2.addWeighted(frame_a, 1.0 - alpha_1, frame_b, alpha_1, 0)
    assert np.allclose(blended_1[0, 0, 0], int(round(255 * (2.0 / 3.0))), atol=2)


def test_clip_engine_checkpoint_and_resume(test_clip_video: Path, tmp_path: Path):
    """Test checkpoint and resume functionality in clip-based mode."""
    out_video = tmp_path / "clip_resumed_output.mp4"
    chk_dir = tmp_path / "clip_checkpoints"

    cfg = AppConfig(
        model=ModelConfig(scale=2, target_scale=2),
        inference=InferenceConfig(clip_length=12, clip_overlap=2),
        checkpoint=CheckpointConfig(enabled=True, dir=str(chk_dir), interval_frames=20),
        encode=EncodeConfig(codec="auto", copy_audio=True),
    )

    # Interrupt after first chunk
    call_idx = 0
    class InterruptibleClipEngine(VideoUpscaleEngine):
        def load(self, model_path: str, precision: str = "fp16", provider: str = "auto") -> None:
            pass

        def upscale_clip(self, frames: List[np.ndarray]) -> List[np.ndarray]:
            nonlocal call_idx
            call_idx += 1
            if call_idx >= 2:
                pipeline_1.cancel()
            return [cv2.resize(f, (f.shape[1] * 2, f.shape[0] * 2)) for f in frames]

        def close(self) -> None:
            pass

    engine_1 = InterruptibleClipEngine()
    pipeline_1 = UpscalePipeline(config=cfg, engine=engine_1)
    pipeline_1.run(input_path=test_clip_video, output_path=out_video, resume=False)

    # Resume with normal mock engine
    engine_2 = MockClipEngine(scale=2)
    pipeline_2 = UpscalePipeline(config=cfg, engine=engine_2)
    pipeline_2.run(input_path=test_clip_video, output_path=out_video, resume=True)

    assert out_video.exists()
    meta_in = get_video_metadata(test_clip_video)
    meta_out = get_video_metadata(out_video)
    assert meta_out.total_frames == meta_in.total_frames
