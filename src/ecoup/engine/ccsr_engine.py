"""CCSR (Content-Consistent Super-Resolution) Engine implementation.

Uses CCSR (IEEE TIP 2024 / CVPR) diffusion prior for high-fidelity super-resolution.
Runs on PyTorch with CUDA acceleration (Google Colab Tesla T4/A100).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image

from ecoup.engine.base import UpscaleEngine


class CcsrUpscaleEngine(UpscaleEngine):
    """CCSR AI Inference Engine using PyTorch and Diffusion Priors."""

    def __init__(self, model_path: Optional[str | Path] = None, device: str = "cuda"):
        self.device = device
        self.scale = 2
        self.pipeline = None
        self.active_provider = "PyTorch CUDA (CCSR Diffusion)"
        if model_path is not None:
            self.load(model_path, provider=device)

    def load(self, model_path: str | Path = "preset/models", precision: str = "fp16", provider: str = "cuda") -> None:
        """Load CCSR diffusion model and initialize pipeline."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch chưa được cài đặt. Vui lòng cài đặt PyTorch và chạy trên GPU.")

        if not torch.cuda.is_available() and provider in ("auto", "cuda"):
            self.device = "cpu"
            self.active_provider = "PyTorch CPU"
        else:
            self.device = "cuda"
            self.active_provider = "CUDA (Tesla T4)"

        dtype = torch.float16 if self.device == "cuda" and precision == "fp16" else torch.float32

        # Locate CCSR repository directory
        ccsr_dirs = [
            Path("ccsr_core"),
            Path("CCSR"),
            Path("../CCSR"),
            Path(__file__).parent / "ccsr",
        ]
        for d in ccsr_dirs:
            if d.exists() and (d / "pipelines").exists():
                resolved = str(d.resolve())
                if resolved not in sys.path:
                    sys.path.insert(0, resolved)
                break

        # Check weights path
        ctrl_path = Path("preset/models/controlnet")
        if not ctrl_path.exists():
            ctrl_path = Path("models/controlnet")

        base_model_id = "stabilityai/stable-diffusion-2-1-base"

        try:
            from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
            from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
        except ImportError:
            raise RuntimeError("Thiếu thư viện diffusers / transformers. Vui lòng chạy lệnh pip install.")

        # Attempt to load ControlNetModel
        controlnet = None
        try:
            from models.controlnet import ControlNetModel
            if ctrl_path.exists():
                controlnet = ControlNetModel.from_pretrained(str(ctrl_path), torch_dtype=dtype)
            else:
                # Try Hugging Face directly
                controlnet = ControlNetModel.from_pretrained("YaronElh/CCSR-v2", torch_dtype=dtype)
        except Exception:
            try:
                from diffusers import ControlNetModel
                controlnet = ControlNetModel.from_pretrained("YaronElh/CCSR-v2", torch_dtype=dtype)
            except Exception as e:
                raise RuntimeError(
                    f"Chưa tìm thấy trọng số CCSR v2 ({e}).\n"
                    "Vui lòng chạy ô [BƯỚC 3B] trong Colab Notebook để tải mã nguồn và mô hình CCSR."
                )

        try:
            from pipelines.pipeline_ccsr import StableDiffusionControlNetPipeline
            pipe_cls = StableDiffusionControlNetPipeline
        except ImportError:
            from diffusers import StableDiffusionControlNetPipeline
            pipe_cls = StableDiffusionControlNetPipeline

        print(f"[CCSR Engine] Đang khởi tạo Stable Diffusion 2.1 Base & CCSR ControlNet trên {self.device}...")
        scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
        feature_extractor = CLIPImageProcessor.from_pretrained(base_model_id, subfolder="feature_extractor")
        unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=dtype)

        pipe = pipe_cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(self.device)
        self.pipeline = pipe
        print(f"[CCSR Engine] Khởi động CCSR Diffusion thành công trên {self.device}!")

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """Upscale RGB frame (H, W, 3) uint8 using CCSR Diffusion."""
        if self.pipeline is None:
            raise RuntimeError("CCSR Pipeline chưa được khởi tạo. Hãy gọi load() trước.")

        import torch
        pil_img = Image.fromarray(frame)
        w, h = pil_img.size
        target_w, target_h = w * self.scale, h * self.scale

        with torch.inference_mode():
            # 1-step efficient diffusion restoration
            res = self.pipeline(
                prompt="",
                image=pil_img,
                num_inference_steps=1,
                guidance_scale=1.0,
                output_type="np",
            )
            out_np = res.images[0]
            if out_np.max() <= 1.0:
                out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)

        # Scale alignment if needed
        if out_np.shape[1] != target_w or out_np.shape[0] != target_h:
            out_np = cv2.resize(out_np, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return out_np

    def close(self):
        self.pipeline = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
