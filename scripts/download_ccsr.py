"""Download script for CCSR v2 diffusion weights from HuggingFace."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def download_ccsr():
    print("=" * 60)
    print("Đang tải trọng số CCSR v2 từ Hugging Face...")
    print("=" * 60)

    dest_dir = Path("preset/models")
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Đang cài đặt huggingface_hub...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    ctrl_dir = dest_dir / "controlnet"
    ctrl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Đang tải CCSR-v2 checkpoint vào: {ctrl_dir}...")
    snapshot_download(
        repo_id="YaronElh/CCSR-v2",
        local_dir=str(ctrl_dir),
        ignore_patterns=["*.md", "*.git*"],
    )
    print("Hoàn tất tải trọng số CCSR v2!")


if __name__ == "__main__":
    download_ccsr()
