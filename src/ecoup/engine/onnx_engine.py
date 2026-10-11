"""ONNX Runtime inference engine with CUDA/TensorRT acceleration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import numpy as np
import onnxruntime as ort

from ecoup.engine.base import UpscaleEngine


class OnnxUpscaleEngine(UpscaleEngine):
    """Inference engine powered by ONNX Runtime."""

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_name: str = ""
        self.precision: str = "fp16"
        self.is_model_fp16: bool = False
        self.scale: int = 4

    def load(self, model_path: str | Path, precision: str = "fp16", provider: str = "auto") -> None:
        model_str = str(Path(model_path).resolve())
        if not os.path.exists(model_str):
            raise FileNotFoundError(f"ONNX model file not found: {model_str}")

        available = ort.get_available_providers()
        chosen_providers: list[str] = []

        if provider == "auto":
            if "DmlExecutionProvider" in available:
                chosen_providers.append("DmlExecutionProvider")
            if "CUDAExecutionProvider" in available:
                chosen_providers.append("CUDAExecutionProvider")
            if "TensorrtExecutionProvider" in available:
                chosen_providers.append("TensorrtExecutionProvider")
            if "CPUExecutionProvider" in available:
                chosen_providers.append("CPUExecutionProvider")
        elif provider in ("dml", "directml"):
            chosen_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif provider == "cuda":
            chosen_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "tensorrt":
            chosen_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "cpu":
            chosen_providers = ["CPUExecutionProvider"]
        else:
            chosen_providers = [provider, "CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_str, sess_options=opts, providers=chosen_providers)
        self.precision = precision.lower()

        # Retrieve input / output metadata
        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]
        self.input_name = input_meta.name
        self.output_name = output_meta.name

        if "float16" in input_meta.type.lower():
            self.input_dtype = np.float16
        else:
            self.input_dtype = np.float32

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """Upscale RGB frame (H, W, 3) uint8 using ONNX session."""
        if self.session is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        # Convert uint8 [0, 255] to normalized float matching model graph input type
        img = frame.astype(self.input_dtype) / self.input_dtype(255.0)

        # HWC (H, W, 3) -> NCHW (1, 3, H, W)
        tensor = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        out_tensor = outputs[0]

        # NCHW -> HWC
        out_img = np.transpose(out_tensor[0], (1, 2, 0))
        out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
        return out_img

    def close(self) -> None:
        self.session = None
