"""Helper script to download verified pre-trained ONNX models for Eco Video Upscaler."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

# Verified public ONNX models
MODELS = {
    "2xNomosUni_compact_otf_medium": {
        "url": "https://huggingface.co/notaneimu/onnx-image-models/resolve/main/2xNomosUni_compact_otf_medium.onnx",
        "desc": "NomosUni Compact 2x (2.3MB) - Siêu nét video & ảnh thực tế, nhẹ và tối ưu",
        "scale": 2,
    },
    "2x-spanx2-ch48": {
        "url": "https://huggingface.co/notaneimu/onnx-image-models/resolve/main/2x-spanx2-ch48.onnx",
        "desc": "SPAN Compact 2x (1.6MB) - Siêu nhanh, mát GPU, tiết kiệm điện năng",
        "scale": 2,
    },
    "2x-realesrgan-x2plus": {
        "url": "https://huggingface.co/notaneimu/onnx-image-models/resolve/main/2x-realesrgan-x2plus.onnx",
        "desc": "Real-ESRGAN x2plus (67MB) - Tái tạo chi tiết & texture sắc sảo cao cấp",
        "scale": 2,
    },
    "realesr-animevideov3": {
        "url": "https://huggingface.co/nimjinwei/designerplatform-models/resolve/main/realesr-animevideov3.onnx",
        "desc": "Real-ESRGAN Anime Video V3 (Compact 4x) - Hoạt hình & Anime 2D",
        "scale": 4,
    },
    "realesr-general-x4v3": {
        "url": "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-general-x4v3.onnx",
        "desc": "Real-ESRGAN General X4 V3 (Compact 4x) - Khử nhiễu mịn da, cảnh đời thực",
        "scale": 4,
    },
}


def download_file(url: str, dest_path: Path) -> None:
    print(f"Downloading from: {url}")
    print(f"Saving to: {dest_path}")
    os.makedirs(dest_path.parent, exist_ok=True)

    def report(count, block_size, total_size):
        percent = int(count * block_size * 100 / max(1, total_size))
        print(f"\rProgress: {percent}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest_path), reporthook=report)
    print("\nDownload finished successfully!")


def main():
    parser = argparse.ArgumentParser(description="Download ONNX models for Eco Video Upscaler")
    parser.add_argument(
        "model",
        choices=list(MODELS.keys()),
        default="realesr-animevideov3",
        nargs="?",
        help="Model name to download",
    )
    args = parser.parse_args()

    info = MODELS[args.model]
    dest = Path("models") / f"{args.model}.onnx"
    print(f"Selected: {args.model} - {info['desc']}")
    download_file(info["url"], dest)
    print(f"Model is ready for inference at: {dest}")


if __name__ == "__main__":
    main()
