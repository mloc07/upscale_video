"""Unit tests for OnnxUpscaleEngine and scale detection."""

import numpy as np
import pytest
from pathlib import Path

from ecoup.engine.onnx_engine import OnnxUpscaleEngine


def test_onnx_engine_load_scale_2x():
    model_path = Path("models/2xNomosUni_compact_otf_medium.onnx")
    if not model_path.exists():
        pytest.skip("Model not downloaded")

    engine = OnnxUpscaleEngine()
    engine.load(model_path, provider="dml")
    assert engine.scale == 2

    # Test upscale
    dummy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    out = engine.upscale(dummy)
    assert out.shape == (128, 128, 3)
    engine.close()


def test_onnx_engine_load_scale_4x():
    model_path = Path("models/realesr-general-x4v3.onnx")
    if not model_path.exists():
        pytest.skip("Model not downloaded")

    engine = OnnxUpscaleEngine()
    engine.load(model_path, provider="dml")
    assert engine.scale == 4

    dummy = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    out = engine.upscale(dummy)
    assert out.shape == (128, 128, 3)
    engine.close()
