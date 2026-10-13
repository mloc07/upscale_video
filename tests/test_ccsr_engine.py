"""Unit tests for CCSR engine native 4x behavior."""

from __future__ import annotations

import types
import sys
import numpy as np
import pytest
from PIL import Image

from ecoup.engine.ccsr_engine import CcsrUpscaleEngine


class _FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeGenerator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class _FakePipeline:
    def __init__(self, out_size: tuple[int, int]):
        self.out_size = out_size
        self.calls = []

    def __call__(
        self,
        t_max,
        t_min,
        tile_diffusion,
        tile_size,
        tile_stride,
        prompt,
        image,
        generator,
        num_inference_steps,
        guidance_scale,
        output_type,
        conditioning_scale,
        start_steps,
        start_point,
        use_vae_encode_condition,
    ):
        self.calls.append(
            {
                "t_max": t_max,
                "t_min": t_min,
                "tile_diffusion": tile_diffusion,
                "tile_size": tile_size,
                "tile_stride": tile_stride,
                "prompt": prompt,
                "image_size": image.size,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "output_type": output_type,
                "conditioning_scale": conditioning_scale,
                "start_steps": start_steps,
                "start_point": start_point,
                "use_vae_encode_condition": use_vae_encode_condition,
            }
        )
        out_pil = Image.new("RGB", self.out_size, color=(12, 34, 56))
        return None, types.SimpleNamespace(images=[out_pil])


@pytest.fixture
def fake_torch(monkeypatch):
    fake_torch_mod = types.SimpleNamespace(
        Generator=_FakeGenerator,
        inference_mode=lambda: _FakeInferenceMode(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch_mod)


def test_ccsr_engine_native_scale_is_4():
    engine = CcsrUpscaleEngine()
    assert engine.scale == 4


def test_upscale_raises_if_ccsr_output_wrong_size(fake_torch):
    engine = CcsrUpscaleEngine()
    engine.is_ccsr_pipeline = True
    engine.pipeline = _FakePipeline(out_size=(15, 11))

    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="CCSR output sai kích thước"):
        engine.upscale(frame)


def test_upscale_accepts_native_4x_output_and_passes_quality_params(fake_torch):
    engine = CcsrUpscaleEngine(control_strength=0.75)
    engine.is_ccsr_pipeline = True
    engine.pipeline = _FakePipeline(out_size=(16, 12))

    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    out = engine.upscale(frame, control_strength=1.1)

    assert out.shape == (12, 16, 3)
    assert np.all(out == np.array([12, 34, 56], dtype=np.uint8))

    assert len(engine.pipeline.calls) == 1
    call = engine.pipeline.calls[0]
    assert call["image_size"] == (16, 12)
    assert call["t_max"] == pytest.approx(0.6667)
    assert call["t_min"] == pytest.approx(0.5)
    assert call["num_inference_steps"] == 6
    assert call["guidance_scale"] == pytest.approx(4.5)
    assert call["conditioning_scale"] == pytest.approx(1.1)
