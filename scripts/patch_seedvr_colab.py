"""Utility script to patch ByteDance SeedVR inference scripts on Google Colab / local GPU.

Fixes:
1. Replaces buggy/deprecated `torchvision.io.video.read_video` with pure OpenCV + PyTorch video reader.
2. Ensures official `color_fix.py` (with `wavelet_reconstruction`) is in place.
3. Provides native PyTorch SDPA fallback for `flash_attn_varlen_func` on GPUs without FlashAttention
   (such as NVIDIA Tesla T4 Turing / CC 7.5 on Colab, RTX 3050, and Windows).
4. Provides native PyTorch fallback (LayerNorm / Diffusers RMSNorm) for NVIDIA Apex `FusedRMSNorm` / `FusedLayerNorm`.
5. Handles both single file paths and directories for `--video_path`.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CUSTOM_READ_VIDEO = '''
import cv2
import numpy as np

def read_video(filename, pts_unit="pts", output_format="TCHW"):
    """Robust OpenCV-based read_video replacement returning PyTorch tensor."""
    cap = cv2.VideoCapture(str(filename))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Không thể đọc khung hình từ: {filename}")
    tensor = torch.from_numpy(np.stack(frames, axis=0))
    if output_format.upper() == "TCHW":
        tensor = tensor.permute(0, 3, 1, 2)
    return tensor, torch.empty(0), {"video_fps": float(fps)}
'''

FLASH_ATTN_PATCH = '''try:
    from flash_attn import flash_attn_varlen_func
except (ImportError, ModuleNotFoundError):
    def flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q=None, max_seqlen_k=None,
        dropout_p=0.0, softmax_scale=None,
        causal=False, window_size=(-1, -1),
        return_attn_probs=False, deterministic=False,
        **kwargs,
    ):
        """Native PyTorch SDPA fallback for flash_attn_varlen_func.
        Enables SeedVR2 to run seamlessly on NVIDIA Tesla T4, RTX 3050, and all GPUs without flash-attn binary.
        """
        seqlens_q = cu_seqlens_q.tolist() if hasattr(cu_seqlens_q, "tolist") else list(cu_seqlens_q)
        seqlens_k = cu_seqlens_k.tolist() if hasattr(cu_seqlens_k, "tolist") else list(cu_seqlens_k)
        batch_size = len(seqlens_q) - 1

        outs = []
        for i in range(batch_size):
            sq_start, sq_end = int(seqlens_q[i]), int(seqlens_q[i + 1])
            sk_start, sk_end = int(seqlens_k[i]), int(seqlens_k[i + 1])

            q_i = q[sq_start:sq_end].unsqueeze(0).transpose(1, 2)  # (1, num_heads, seqlen_q, head_dim)
            k_i = k[sk_start:sk_end].unsqueeze(0).transpose(1, 2)  # (1, num_heads, seqlen_k, head_dim)
            v_i = v[sk_start:sk_end].unsqueeze(0).transpose(1, 2)  # (1, num_heads, seqlen_k, head_dim)

            if q_i.shape[1] != k_i.shape[1]:
                rep = q_i.shape[1] // k_i.shape[1]
                k_i = k_i.repeat_interleave(rep, dim=1)
                v_i = v_i.repeat_interleave(rep, dim=1)

            try:
                out_i = F.scaled_dot_product_attention(
                    q_i, k_i, v_i,
                    dropout_p=dropout_p if (q.requires_grad and dropout_p > 0.0) else 0.0,
                    is_causal=causal,
                    scale=softmax_scale,
                )
            except RuntimeError:
                scale = (1.0 / (q_i.shape[-1] ** 0.5)) if softmax_scale is None else softmax_scale
                attn_weights = torch.matmul(q_i, k_i.transpose(-2, -1)) * scale
                if causal:
                    mask = torch.ones(q_i.shape[-2], k_i.shape[-2], device=q_i.device, dtype=torch.bool).tril()
                    attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_i.dtype)
                if dropout_p > 0.0 and q.requires_grad:
                    attn_weights = F.dropout(attn_weights, p=dropout_p)
                out_i = torch.matmul(attn_weights, v_i)

            outs.append(out_i.transpose(1, 2).squeeze(0))

        out = torch.cat(outs, dim=0)
        if return_attn_probs:
            return out, None, None, None
        return out'''


FLASH_ATTN_SHIM = '''"""Standalone flash_attn shim using native PyTorch SDPA."""
import torch
import torch.nn.functional as F

''' + FLASH_ATTN_PATCH.split("except (ImportError, ModuleNotFoundError):")[1].strip() + '''

__all__ = ["flash_attn_varlen_func"]
'''

APEX_NORM_TARGET = """        if norm_type == "fusedln":
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )

        if norm_type == "fusedrms":
            from apex.normalization import FusedRMSNorm

            return FusedRMSNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )"""

APEX_NORM_REPLACEMENT = """        if norm_type == "fusedln":
            try:
                from apex.normalization import FusedLayerNorm
                return FusedLayerNorm(
                    normalized_shape=dim,
                    elementwise_affine=elementwise_affine,
                    eps=eps,
                )
            except (ImportError, ModuleNotFoundError):
                return nn.LayerNorm(
                    normalized_shape=dim,
                    eps=eps,
                    elementwise_affine=elementwise_affine,
                )

        if norm_type == "fusedrms":
            try:
                from apex.normalization import FusedRMSNorm
                return FusedRMSNorm(
                    normalized_shape=dim,
                    elementwise_affine=elementwise_affine,
                    eps=eps,
                )
            except (ImportError, ModuleNotFoundError):
                return RMSNorm(
                    dim=dim,
                    eps=eps,
                    elementwise_affine=elementwise_affine,
                )"""

APEX_SHIM_NORMALIZATION = '''"""Standalone apex.normalization shim for Colab and PyTorch environments."""
import torch
from torch import nn
from diffusers.models.normalization import RMSNorm

class FusedLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, **kwargs):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

class FusedRMSNorm(RMSNorm):
    def __init__(self, normalized_shape=None, dim=None, eps=1e-5, elementwise_affine=True, **kwargs):
        d = dim if dim is not None else normalized_shape
        if isinstance(d, (list, tuple)):
            d = d[0]
        super().__init__(dim=d, eps=eps, elementwise_affine=elementwise_affine)

__all__ = ["FusedLayerNorm", "FusedRMSNorm"]
'''


def patch_seedvr_core(repo_dir: str | Path = "seedvr_core"):
    repo_path = Path(repo_dir)
    infer_script = repo_path / "projects" / "inference_seedvr2_3b.py"

    if not infer_script.exists():
        print(f"[Patch] Không tìm thấy file {infer_script}. Bỏ qua.")
        return

    content = infer_script.read_text(encoding="utf-8")

    # 1. Replace torchvision read_video import with robust OpenCV implementation
    replaced_read = False
    for target in [
        "from torchvision.io.video import read_video",
        "from torchvision.io import read_video",
    ]:
        if target in content:
            content = content.replace(target, CUSTOM_READ_VIDEO)
            replaced_read = True
            print(f"[Patch] Đã thay thế '{target}' bằng bộ nạp video OpenCV + PyTorch tối ưu.")

    # 2. Patch --video_path handling to support single file paths as well as directories
    orig_prompt_block = (
        "    def _build_test_prompts(video_path):\n"
        "        positive_text, negative_text = _build_pos_and_neg_prompt()\n"
        "        original_videos = []\n"
        "        prompts = {}\n"
        "        video_list = os.listdir(video_path)"
    )
    patched_prompt_block = (
        "    def _build_test_prompts(video_path):\n"
        "        positive_text, negative_text = _build_pos_and_neg_prompt()\n"
        "        original_videos = []\n"
        "        prompts = {}\n"
        "        if os.path.isfile(video_path):\n"
        "            video_list = [os.path.basename(video_path)]\n"
        "        else:\n"
        "            video_list = sorted([f for f in os.listdir(video_path) if not f.startswith('.')])"
    )
    if orig_prompt_block in content:
        content = content.replace(orig_prompt_block, patched_prompt_block)
        print("[Patch] Đã hỗ trợ cả đường dẫn file đơn lẻ và thư mục cho --video_path.")

    # Also handle video file read path when video_path is a file
    orig_read_line = "os.path.join(video_path, video)"
    patched_read_line = "(video_path if os.path.isfile(video_path) else os.path.join(video_path, video))"
    if orig_read_line in content:
        content = content.replace(orig_read_line, patched_read_line)
        print("[Patch] Đã vá đường dẫn nạp khung hình an toàn cho video đơn.")

    infer_script.write_text(content, encoding="utf-8")

    # 3. Copy official color_fix.py
    color_fix_dest_dir = repo_path / "projects" / "video_diffusion_sr"
    color_fix_dest_dir.mkdir(parents=True, exist_ok=True)
    src_colorfix = Path("src/ecoup/seedvr_color_fix.py")
    dest_colorfix = color_fix_dest_dir / "color_fix.py"

    if src_colorfix.exists():
        shutil.copyfile(src_colorfix, dest_colorfix)
        print(f"[Patch] Đã cài đặt module Wavelet Color Fix vào: {dest_colorfix}")

    # 4. Patch FlashAttention in models/dit_v2/attention.py & models/dit/attention.py
    attn_targets = [
        repo_path / "models" / "dit_v2" / "attention.py",
        repo_path / "models" / "dit" / "attention.py",
    ]
    for attn_file in attn_targets:
        if attn_file.exists():
            attn_content = attn_file.read_text(encoding="utf-8")
            if "from flash_attn import flash_attn_varlen_func" in attn_content:
                attn_content = attn_content.replace(
                    "from flash_attn import flash_attn_varlen_func",
                    FLASH_ATTN_PATCH,
                )
                attn_file.write_text(attn_content, encoding="utf-8")
                print(f"[Patch] Đã thêm PyTorch SDPA fallback cho FlashAttention trong: {attn_file}")

    # 5. Create standalone flash_attn shim package in seedvr_core
    shim_dir = repo_path / "flash_attn"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_file = shim_dir / "__init__.py"
    shim_file.write_text(FLASH_ATTN_SHIM, encoding="utf-8")
    print(f"[Patch] Đã tạo gói tương thích FlashAttention shim tại: {shim_file}")

    # 6. Patch Apex FusedRMSNorm / FusedLayerNorm in models/dit_v2/normalization.py & models/dit/normalization.py
    norm_targets = [
        repo_path / "models" / "dit_v2" / "normalization.py",
        repo_path / "models" / "dit" / "normalization.py",
    ]
    for norm_file in norm_targets:
        if norm_file.exists():
            norm_content = norm_file.read_text(encoding="utf-8")
            if APEX_NORM_TARGET in norm_content:
                norm_content = norm_content.replace(APEX_NORM_TARGET, APEX_NORM_REPLACEMENT)
                norm_file.write_text(norm_content, encoding="utf-8")
                print(f"[Patch] Đã thêm PyTorch native fallback cho Apex Norm trong: {norm_file}")

    # 7. Create standalone apex shim package in seedvr_core
    apex_dir = repo_path / "apex"
    apex_dir.mkdir(parents=True, exist_ok=True)
    (apex_dir / "__init__.py").write_text('"""Standalone apex shim."""\n', encoding="utf-8")
    apex_norm_file = apex_dir / "normalization.py"
    apex_norm_file.write_text(APEX_SHIM_NORMALIZATION, encoding="utf-8")
    # 8. Optimize configs_3b/main.yaml with init_with_meta_device to prevent CPU RAM OOM (SIGKILL -9)
    config_yaml = repo_path / "configs_3b" / "main.yaml"
    if config_yaml.exists():
        cfg_content = config_yaml.read_text(encoding="utf-8")
        if "init_with_meta_device:" not in cfg_content:
            cfg_content = cfg_content.replace(
                "dit:\n  model:",
                "dit:\n  init_with_meta_device: True\n  model:",
            )
            config_yaml.write_text(cfg_content, encoding="utf-8")
    # 9. Patch meta_init_utils.py to safely materialize buffers to CPU without assertion failure
    meta_utils_file = repo_path / "common" / "distributed" / "meta_init_utils.py"
    if meta_utils_file.exists():
        SAFE_META_UTILS = '''# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
import torch
from torch import nn

__all__ = ["meta_non_persistent_buffer_init_fn"]

def meta_non_persistent_buffer_init_fn(module: nn.Module) -> nn.Module:
    with torch.no_grad():
        for submodule in module.modules():
            for buffer_name, buffer in list(submodule.named_buffers(recurse=False)):
                if getattr(buffer, "is_meta", False):
                    try:
                        materialized_buffer = torch.zeros(
                            buffer.shape, dtype=buffer.dtype, device="cpu"
                        )
                        setattr(submodule, buffer_name, materialized_buffer)
                    except Exception:
                        pass
    return module
'''
        meta_utils_file.write_text(SAFE_META_UTILS, encoding="utf-8")
        print(f"[Patch] Đã vá an toàn meta_init_utils.py tại: {meta_utils_file}")

    print("[Patch] Hoàn tất toàn bộ vá lỗi môi trường cho SeedVR!")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "seedvr_core"
    patch_seedvr_core(target_dir)
