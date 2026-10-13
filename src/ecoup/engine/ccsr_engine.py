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
from ecoup.color_fix import adain_color_fix


class CcsrUpscaleEngine(UpscaleEngine):
    """CCSR AI Inference Engine using PyTorch and Diffusion Priors."""

    def __init__(self, model_path: Optional[str | Path] = None, device: str = "cuda"):
        self.device = device
        self.scale = 2
        self.pipeline = None
        self.is_ccsr_pipeline = False
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

        # Locate CCSR repository directory and add to sys.path
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

        # Locate local controlnet checkpoint with config.json
        local_ctrl = None
        ctrl_candidates = [
            Path("preset/models/controlnet/controlnet"),
            Path("preset/models/controlnet"),
            Path("models/controlnet/controlnet"),
            Path("models/controlnet"),
        ]
        for c in ctrl_candidates:
            if (c / "config.json").exists():
                local_ctrl = str(c)
                break

        # Locate local custom VAE checkpoint with config.json
        local_vae = None
        vae_candidates = [
            Path("preset/models/controlnet/vae"),
            Path("preset/models/vae"),
            Path("models/vae"),
        ]
        for v in vae_candidates:
            if (v / "config.json").exists():
                local_vae = str(v)
                break

        base_model_id = "Manojb/stable-diffusion-2-1-base"

        try:
            from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
            from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
        except ImportError:
            raise RuntimeError("Thiếu thư viện diffusers / transformers. Vui lòng chạy lệnh pip install.")

        # Load ControlNet model
        controlnet = None
        try:
            try:
                from models.controlnet import ControlNetModel
            except ImportError:
                from diffusers import ControlNetModel

            if local_ctrl:
                print(f"[CCSR Engine] Nạp ControlNet từ thư mục cục bộ: {local_ctrl}")
                try:
                    controlnet = ControlNetModel.from_pretrained(local_ctrl, torch_dtype=dtype, low_cpu_mem_usage=False)
                except TypeError:
                    controlnet = ControlNetModel.from_pretrained(local_ctrl, torch_dtype=dtype)
            else:
                print("[CCSR Engine] Đang tải ControlNet CCSR-v2 từ Hugging Face (YaronElh/CCSR-v2, subfolder='controlnet')...")
                try:
                    controlnet = ControlNetModel.from_pretrained("YaronElh/CCSR-v2", subfolder="controlnet", torch_dtype=dtype, low_cpu_mem_usage=False)
                except TypeError:
                    controlnet = ControlNetModel.from_pretrained("YaronElh/CCSR-v2", subfolder="controlnet", torch_dtype=dtype)
        except Exception as e:
            raise RuntimeError(
                f"Lỗi khi nạp CCSR ControlNet: {e}\n"
                "Vui lòng đảm bảo đã kết nối mạng hoặc chạy ô [BƯỚC 3B] trong Colab Notebook."
            )

        # Load VAE model
        vae = None
        try:
            if local_vae:
                print(f"[CCSR Engine] Nạp custom VAE từ: {local_vae}")
                try:
                    vae = AutoencoderKL.from_pretrained(local_vae, torch_dtype=dtype, low_cpu_mem_usage=False)
                except TypeError:
                    vae = AutoencoderKL.from_pretrained(local_vae, torch_dtype=dtype)
            else:
                print("[CCSR Engine] Đang tải VAE CCSR-v2 từ Hugging Face (YaronElh/CCSR-v2, subfolder='vae')...")
                try:
                    vae = AutoencoderKL.from_pretrained("YaronElh/CCSR-v2", subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=False)
                except TypeError:
                    vae = AutoencoderKL.from_pretrained("YaronElh/CCSR-v2", subfolder="vae", torch_dtype=dtype)
        except Exception as e:
            print(f"[CCSR Engine] Cảnh báo nạp VAE tùy biến ({e}). Sử dụng VAE chuẩn của SD 2.1.")
            try:
                vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=False)
            except TypeError:
                vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=dtype)

        # Pipeline class resolution
        pipe_cls = None
        try:
            from pipelines.pipeline_ccsr import StableDiffusionControlNetPipeline
            pipe_cls = StableDiffusionControlNetPipeline
            self.is_ccsr_pipeline = True
        except ImportError:
            from diffusers import StableDiffusionControlNetPipeline
            pipe_cls = StableDiffusionControlNetPipeline
            self.is_ccsr_pipeline = False

        # Base model candidates (using open public mirrors to prevent 401 Client Error)
        if Path("preset/models/stable-diffusion-2-1-base").exists():
            preferred_base = str(Path("preset/models/stable-diffusion-2-1-base").resolve())
        else:
            preferred_base = "Manojb/stable-diffusion-2-1-base"

        base_candidates = [
            preferred_base,
            "Manojb/stable-diffusion-2-1-base",
            "sd2-community/stable-diffusion-2-1-base",
        ]

        scheduler = None
        for cand in base_candidates:
            try:
                print(f"[CCSR Engine] Đang nạp Stable Diffusion 2.1 Base components từ {cand} trên {self.device}...")
                scheduler = DDPMScheduler.from_pretrained(cand, subfolder="scheduler")
                try:
                    text_encoder = CLIPTextModel.from_pretrained(cand, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=False)
                except TypeError:
                    text_encoder = CLIPTextModel.from_pretrained(cand, subfolder="text_encoder", torch_dtype=dtype)

                tokenizer = CLIPTokenizer.from_pretrained(cand, subfolder="tokenizer")
                feature_extractor = CLIPImageProcessor.from_pretrained(cand, subfolder="feature_extractor")

                try:
                    unet = UNet2DConditionModel.from_pretrained(cand, subfolder="unet", torch_dtype=dtype, low_cpu_mem_usage=False)
                except TypeError:
                    unet = UNet2DConditionModel.from_pretrained(cand, subfolder="unet", torch_dtype=dtype)

                base_model_id = cand
                break
            except Exception as e:
                print(f"[CCSR Engine] Lỗi khi nạp từ {cand} ({e}), đang thử mirror khác...")
                continue

        if scheduler is None:
            raise RuntimeError(
                "Không thể nạp thành phần Stable Diffusion 2.1 Base từ kho lưu trữ.\n"
                "Vui lòng kiểm tra kết nối internet của Google Colab."
            )

        # Move constituent models to device and freeze weights (matching official CCSR test_ccsr_tile.py)
        for m in [vae, text_encoder, unet, controlnet]:
            if m is not None:
                try:
                    m.requires_grad_(False)
                except Exception:
                    pass
                m.to(self.device, dtype=dtype)

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

        gen = torch.Generator(device=self.device).manual_seed(42)

        with torch.inference_mode():
            if self.is_ccsr_pipeline:
                # Official CCSR-v2 pipeline call
                # Signature: (t_max, t_min, tile_diffusion, tile_size, tile_stride, prompt, image, ...)
                _, res = self.pipeline(
                    0.6667,  # t_max
                    0.3333,  # t_min
                    False,   # tile_diffusion (processed frame by frame)
                    512,     # tile_size
                    256,     # tile_stride
                    prompt="",
                    image=pil_img,
                    generator=gen,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    start_steps=999,
                    start_point="lr",
                    use_vae_encode_condition=True,
                    output_type="pil",
                )
                out_pil = res.images[0]
            else:
                # Standard Diffusers fallback call
                res = self.pipeline(
                    prompt="",
                    image=pil_img,
                    generator=gen,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    output_type="pil",
                )
                out_pil = res.images[0]

        out_np = np.array(out_pil)

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
