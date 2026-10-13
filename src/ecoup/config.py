"""Configuration loading and validation for Eco Video Upscaler."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclass
class ModelConfig:
    name: str = "realesr-general-x4v3"
    path: str = "models/realesr-general-x4v3.onnx"
    scale: int = 4
    target_scale: int = 2
    precision: str = "fp16"

    def validate(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("model.name must be a non-empty string.")
        if self.scale <= 0:
            raise ValueError(f"model.scale must be positive, got {self.scale}.")
        if self.target_scale <= 0:
            raise ValueError(f"model.target_scale must be positive, got {self.target_scale}.")
        if self.precision.lower() not in ("fp16", "fp32"):
            raise ValueError(f"model.precision must be 'fp16' or 'fp32', got '{self.precision}'.")


@dataclass
class InferenceConfig:
    provider: str = "auto"
    tile_size: int = 256
    tile_overlap: int = 16
    batch_size: int = 1
    control_strength: float = 0.75   # 0.0 - 2.0 (ControlNet conditioning scale; 0.7-0.8 for crisp textures, 1.0 for strict adherence)

    def validate(self) -> None:
        valid_providers = ("auto", "cuda", "tensorrt", "cpu")
        if self.provider.lower() not in valid_providers:
            raise ValueError(f"inference.provider must be one of {valid_providers}, got '{self.provider}'.")
        if self.tile_size < 0:
            raise ValueError(f"inference.tile_size must be >= 0 (0 for no tiling), got {self.tile_size}.")
        if self.tile_overlap < 0:
            raise ValueError(f"inference.tile_overlap must be >= 0, got {self.tile_overlap}.")
        if self.tile_size > 0 and self.tile_overlap >= self.tile_size // 2:
            raise ValueError(
                f"inference.tile_overlap ({self.tile_overlap}) must be less than half of tile_size ({self.tile_size})."
            )
        if self.batch_size <= 0:
            raise ValueError(f"inference.batch_size must be >= 1, got {self.batch_size}.")
        if not (0.0 <= self.control_strength <= 2.0):
            raise ValueError(f"inference.control_strength must be in range [0.0, 2.0], got {self.control_strength}.")


@dataclass
class FrameSavingConfig:
    enable_dedup: bool = True
    dedup_threshold: float = 0.5
    enable_scene_detect: bool = False

    def validate(self) -> None:
        if not isinstance(self.enable_dedup, bool):
            raise ValueError("frame_saving.enable_dedup must be a boolean.")
        if self.dedup_threshold < 0:
            raise ValueError(f"frame_saving.dedup_threshold must be >= 0, got {self.dedup_threshold}.")


@dataclass
class ThermalConfig:
    enabled: bool = True
    poll_interval_sec: float = 2.0
    temp_soft_c: int = 70
    temp_hard_c: int = 80
    temp_resume_c: int = 65
    max_gpu_util_pct: int = 85
    throttle_sleep_ms_min: int = 0
    throttle_sleep_ms_max: int = 500
    power_limit_watts: Optional[int] = None

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("thermal.enabled must be a boolean.")
        if self.poll_interval_sec <= 0:
            raise ValueError(f"thermal.poll_interval_sec must be > 0, got {self.poll_interval_sec}.")
        if self.temp_soft_c >= self.temp_hard_c:
            raise ValueError(
                f"thermal.temp_soft_c ({self.temp_soft_c}) must be less than temp_hard_c ({self.temp_hard_c})."
            )
        if self.temp_resume_c > self.temp_soft_c:
            raise ValueError(
                f"thermal.temp_resume_c ({self.temp_resume_c}) must be <= temp_soft_c ({self.temp_soft_c})."
            )
        if not (0 < self.max_gpu_util_pct <= 100):
            raise ValueError(f"thermal.max_gpu_util_pct must be in range (0, 100], got {self.max_gpu_util_pct}.")
        if self.throttle_sleep_ms_min < 0 or self.throttle_sleep_ms_max < self.throttle_sleep_ms_min:
            raise ValueError(
                f"Invalid throttle sleep range: [{self.throttle_sleep_ms_min}, {self.throttle_sleep_ms_max}]."
            )
        if self.power_limit_watts is not None and self.power_limit_watts <= 0:
            raise ValueError(f"thermal.power_limit_watts must be positive if set, got {self.power_limit_watts}.")


@dataclass
class EncodeConfig:
    codec: str = "auto"
    crf_or_cq: int = 21
    preset: str = "p4"
    copy_audio: bool = True

    def validate(self) -> None:
        valid_codecs = ("auto", "h264_nvenc", "hevc_nvenc", "libx264")
        if self.codec.lower() not in valid_codecs:
            raise ValueError(f"encode.codec must be one of {valid_codecs}, got '{self.codec}'.")
        if not (0 <= self.crf_or_cq <= 51):
            raise ValueError(f"encode.crf_or_cq must be between 0 and 51, got {self.crf_or_cq}.")


@dataclass
class CheckpointConfig:
    enabled: bool = True
    dir: str = ".ecoup_state"
    interval_frames: int = 300

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("checkpoint.enabled must be a boolean.")
        if not self.dir:
            raise ValueError("checkpoint.dir cannot be empty.")
        if self.interval_frames <= 0:
            raise ValueError(f"checkpoint.interval_frames must be positive, got {self.interval_frames}.")


@dataclass
class PreProcessConfig:
    denoise: float = 0.0      # 0.0 - 1.0 (Edge-preserving bilateral filter; default 0.0 to prevent softening)

    def validate(self) -> None:
        if not (0.0 <= self.denoise <= 1.0):
            raise ValueError(f"pre_process.denoise must be between 0.0 and 1.0, got {self.denoise}.")


@dataclass
class PostProcessConfig:
    color_fix: bool = True           # Wavelet frequency color anchoring to match original luminance & prevent darkening
    recover_detail: float = 0.0      # 0.0 - 0.5 (blend original Lanczos; default 0.0 to prevent blurring AI details)
    sharpen: float = 0.25            # 0.0 - 1.0 (unsharp mask edge enhancement for crisp definition)
    film_grain: float = 0.0          # 0.0 - 0.05 (natural photonic grain)

    def validate(self) -> None:
        if not isinstance(self.color_fix, bool):
            raise ValueError("post_process.color_fix must be a boolean.")
        if not (0.0 <= self.recover_detail <= 1.0):
            raise ValueError(f"post_process.recover_detail must be between 0.0 and 1.0, got {self.recover_detail}.")
        if not (0.0 <= self.sharpen <= 2.0):
            raise ValueError(f"post_process.sharpen must be between 0.0 and 2.0, got {self.sharpen}.")
        if not (0.0 <= self.film_grain <= 0.2):
            raise ValueError(f"post_process.film_grain must be between 0.0 and 0.2, got {self.film_grain}.")


@dataclass
class AppConfig:
    input: Optional[str] = None
    output: Optional[str] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    frame_saving: FrameSavingConfig = field(default_factory=FrameSavingConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    pre_process: PreProcessConfig = field(default_factory=PreProcessConfig)
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)

    def validate(self) -> None:
        self.model.validate()
        self.inference.validate()
        self.frame_saving.validate()
        self.thermal.validate()
        self.encode.validate()
        self.checkpoint.validate()
        self.pre_process.validate()
        self.post_process.validate()


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load and validate configuration from YAML file or return defaults."""
    raw: dict[str, Any] = {}
    
    path_to_load = config_path or "config.yaml"
    if os.path.exists(path_to_load):
        with open(path_to_load, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw = loaded

    # Build dataclasses from dict
    model_cfg = ModelConfig(**raw.get("model", {}))
    inference_cfg = InferenceConfig(**raw.get("inference", {}))
    frame_saving_cfg = FrameSavingConfig(**raw.get("frame_saving", {}))
    thermal_cfg = ThermalConfig(**raw.get("thermal", {}))
    encode_cfg = EncodeConfig(**raw.get("encode", {}))
    checkpoint_cfg = CheckpointConfig(**raw.get("checkpoint", {}))
    pre_process_cfg = PreProcessConfig(**raw.get("pre_process", {}))
    post_process_cfg = PostProcessConfig(**raw.get("post_process", {}))

    app_cfg = AppConfig(
        input=raw.get("input"),
        output=raw.get("output"),
        model=model_cfg,
        inference=inference_cfg,
        frame_saving=frame_saving_cfg,
        thermal=thermal_cfg,
        encode=encode_cfg,
        checkpoint=checkpoint_cfg,
        pre_process=pre_process_cfg,
        post_process=post_process_cfg,
    )

    app_cfg.validate()
    return app_cfg
