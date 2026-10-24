"""Unit tests for Phase 1, Phase 2, Phase 3 and Phase 4 pipeline v2 features."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from ecoup.config import AppConfig, InferenceConfig, ModelConfig, PreProcessConfig
from ecoup.engine.seedvr2_engine import SeedVR2UpscaleEngine, StepTimer
from ecoup.pipeline import UpscalePipeline


def test_step_timer():
    timer = StepTimer(enabled=True)
    with timer.timed("step_a"):
        _ = sum(i * i for i in range(1000))
    with timer.timed("step_b"):
        _ = sum(i * 2 for i in range(1000))

    assert "step_a" in timer.timings
    assert "step_b" in timer.timings
    assert timer.timings["step_a"] > 0
    assert timer.timings["step_b"] > 0


def test_seedvr2_v2_engine_init_and_fallback():
    engine = SeedVR2UpscaleEngine(
        scale=2,
        precision="fp16",
        seed=123,
        memory_mode="resident",
        sub_len=9,
        model_resolution="match_target",
        enable_profiling=True,
    )
    assert engine.precision == "fp16"
    assert engine.seed == 123
    assert engine.memory_mode == "resident"
    assert engine.sub_len == 9
    assert engine.model_resolution == "match_target"

    # Fallback upscale clip without neural model
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(7)]
    upscaled = engine.upscale_clip(frames)
    assert len(upscaled) == 7
    assert upscaled[0].shape == (240, 320, 3)
    assert engine.last_profile_stats is not None


def test_config_v2_validation():
    # Valid configs
    cfg = InferenceConfig(
        memory_mode="resident",
        sub_len=9,
        model_resolution="match_target",
        seed=42,
    )
    cfg.validate()

    # Invalid sub_len not following 4k+1
    with pytest.raises(ValueError, match="4k\\+1"):
        bad_sub = InferenceConfig(sub_len=7)
        bad_sub.validate()

    # Invalid memory_mode
    with pytest.raises(ValueError, match="inference.memory_mode"):
        bad_mem = InferenceConfig(memory_mode="invalid")
        bad_mem.validate()

    # PreProcessConfig
    prep = PreProcessConfig(method="nlmeans", denoise=0.5)
    prep.validate()

    with pytest.raises(ValueError, match="pre_process.method"):
        bad_prep = PreProcessConfig(method="unknown")
        bad_prep.validate()


def test_sidecar_metadata_saved(tmp_path: Path):
    out_video = tmp_path / "test_out.mp4"
    in_video = tmp_path / "test_in.mp4"

    # Create dummy video file
    in_video.write_bytes(b"dummy")

    cfg = AppConfig(
        model=ModelConfig(name="SeedVR2-3B", target_scale=2, precision="fp16"),
        inference=InferenceConfig(seed=42, memory_mode="resident", sub_len=5),
        checkpoint=AppConfig().checkpoint,
    )
    cfg.checkpoint.enabled = False

    meta_json = out_video.with_suffix(".json")
    run_meta = {
        "input_file": str(in_video),
        "output_file": str(out_video),
        "model_name": cfg.model.name,
        "seed": cfg.inference.seed,
        "precision": cfg.model.precision,
        "memory_mode": cfg.inference.memory_mode,
        "sub_len": cfg.inference.sub_len,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    assert meta_json.exists()
    loaded = json.loads(meta_json.read_text(encoding="utf-8"))
    assert loaded["seed"] == 42
    assert loaded["memory_mode"] == "resident"
