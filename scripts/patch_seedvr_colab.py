"""Utility script to patch ByteDance SeedVR inference scripts on Google Colab.

Fixes:
1. Replaces buggy/deprecated `torchvision.io.video.read_video` with pure OpenCV + PyTorch video reader.
2. Ensures official `color_fix.py` (with `wavelet_reconstruction`) is in place.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

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


def patch_seedvr_core(repo_dir: str | Path = "seedvr_core"):
    repo_path = Path(repo_dir)
    infer_script = repo_path / "projects" / "inference_seedvr2_3b.py"

    if not infer_script.exists():
        print(f"[Patch] Không tìm thấy file {infer_script}. Bỏ qua.")
        return

    content = infer_script.read_text(encoding="utf-8")

    # 1. Replace torchvision read_video import with robust OpenCV implementation
    replaced = False
    for target in [
        "from torchvision.io.video import read_video",
        "from torchvision.io import read_video",
    ]:
        if target in content:
            content = content.replace(target, CUSTOM_READ_VIDEO)
            replaced = True
            print(f"[Patch] Đã thay thế '{target}' bằng bộ nạp video OpenCV + PyTorch tối ưu.")

    if replaced:
        infer_script.write_text(content, encoding="utf-8")

    # 2. Copy official color_fix.py
    color_fix_dest_dir = repo_path / "projects" / "video_diffusion_sr"
    color_fix_dest_dir.mkdir(parents=True, exist_ok=True)
    src_colorfix = Path("src/ecoup/seedvr_color_fix.py")
    dest_colorfix = color_fix_dest_dir / "color_fix.py"

    if src_colorfix.exists():
        shutil.copyfile(src_colorfix, dest_colorfix)
        print(f"[Patch] Đã cài đặt module Wavelet Color Fix vào: {dest_colorfix}")

    print("[Patch] Hoàn tất vá lỗi môi trường cho SeedVR!")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "seedvr_core"
    patch_seedvr_core(target_dir)
