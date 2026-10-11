"""Unit tests for configuration loading and validation."""

import pytest
from pathlib import Path
from ecoup.config import AppConfig, ModelConfig, ThermalConfig, InferenceConfig, load_config


def test_default_config_valid():
    cfg = AppConfig()
    cfg.validate()
    assert cfg.model.scale == 4
    assert cfg.model.target_scale == 2
    assert cfg.thermal.temp_soft_c == 70
    assert cfg.thermal.temp_hard_c == 80


def test_invalid_precision_raises():
    cfg = AppConfig(model=ModelConfig(precision="invalid"))
    with pytest.raises(ValueError, match="model.precision"):
        cfg.validate()


def test_invalid_thermal_temperatures_raise():
    # soft temp must be strictly less than hard temp
    cfg = AppConfig(thermal=ThermalConfig(temp_soft_c=85, temp_hard_c=80))
    with pytest.raises(ValueError, match="temp_soft_c .* must be less than temp_hard_c"):
        cfg.validate()


def test_invalid_tile_overlap_raises():
    # overlap >= tile_size // 2 is invalid
    cfg = AppConfig(inference=InferenceConfig(tile_size=256, tile_overlap=128))
    with pytest.raises(ValueError, match="tile_overlap .* must be less than half of tile_size"):
        cfg.validate()


def test_load_config_from_file(tmp_path: Path):
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text(
        """
model:
  name: test-model
  scale: 2
  target_scale: 2
  precision: fp32
thermal:
  temp_soft_c: 65
  temp_hard_c: 75
  temp_resume_c: 60
""",
        encoding="utf-8",
    )
    loaded = load_config(cfg_file)
    assert loaded.model.name == "test-model"
    assert loaded.model.scale == 2
    assert loaded.model.precision == "fp32"
    assert loaded.thermal.temp_soft_c == 65
