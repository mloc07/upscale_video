"""Download script for SeedVR2-3B weights from HuggingFace."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def download_seedvr2():
    print("=" * 60)
    print("Đang tải trọng số SeedVR2-3B từ Hugging Face...")
    print("=" * 60)

    dest_dir = Path("models/seedvr2_3b")
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Đang cài đặt huggingface_hub...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    print(f"Đang tải SeedVR2-3B checkpoint vào: {dest_dir}...")
    snapshot_download(
        repo_id="ByteDance-Seed/SeedVR2-3B",
        local_dir=str(dest_dir),
        ignore_patterns=["*.git*"],
        resume_download=True,
    )
    print("Hoàn tất tải trọng số SeedVR2-3B!")


if __name__ == "__main__":
    download_seedvr2()
