"""A/B Testing and Preset Optimization for Video Super-Resolution Pre/Post-Processing.

Tests the quantitative (PSNR, SSIM, Delta-E, Edge Sharpness) and qualitative
impact of:
1. Pre-processing: Edge-preserving noise suppression (denoise).
2. Color anchoring: Wavelet frequency color stabilization (color_fix).
3. Identity preservation: Subtle blending with resized original (recover_detail).
4. Subtle sharpening: Unsharp Mask / CAS (sharpen).
5. Natural skin texture: Photonic film grain to prevent plastic look (film_grain).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ecoup.color_fix import wavelet_color_fix
from ecoup.video_io import VideoReader, get_video_metadata

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    y1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(y1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(y2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(y1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(y2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(y1 * y2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12)
    return float(np.clip(ssim_map.mean(), -1.0, 1.0))


def calculate_color_delta_e(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Delta-E in CIE-Lab color space between two images."""
    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))
    return float(np.mean(diff))


def calculate_laplacian_sharpness(img: np.ndarray) -> float:
    """Laplacian variance measure for edge sharpness."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def apply_pipeline_stages(
    raw_frame: np.ndarray,
    upscaled_frame: np.ndarray,
    denoise: float = 0.0,
    color_fix: bool = False,
    recover_detail: float = 0.0,
    sharpen: float = 0.0,
    film_grain: float = 0.0,
) -> np.ndarray:
    """Apply pre and post processing modules on upscaled frame."""
    out = upscaled_frame.copy()
    out_h, out_w = out.shape[:2]

    # Pre-process simulation if denoise > 0 (bilateral edge-preserving)
    if denoise > 0.0:
        sigma = 15.0 + denoise * 45.0
        denoised = cv2.bilateralFilter(out, d=5, sigmaColor=sigma, sigmaSpace=sigma)
        out = cv2.addWeighted(denoised, denoise, out, 1.0 - denoise, 0)

    # Post-process stage 1: Wavelet Frequency Color Anchoring
    if color_fix:
        out = wavelet_color_fix(out, raw_frame)

    # Post-process stage 2: Recover Original Detail (Alpha blend with Lanczos/Bicubic)
    if recover_detail > 0.0:
        base = cv2.resize(raw_frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        out = cv2.addWeighted(out, 1.0 - recover_detail, base, recover_detail, 0)

    # Post-process stage 3: Adaptive Sharpening (Unsharp Mask)
    if sharpen > 0.0:
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.5)
        out = cv2.addWeighted(out, 1.0 + sharpen, blurred, -sharpen, 0)

    # Post-process stage 4: Film Grain (Photonic micro-texture)
    if film_grain > 0.0:
        np.random.seed(42)  # Deterministic seed for reproducible testing
        noise = np.random.normal(0, film_grain * 255.0, (out_h, out_w, 1)).astype(np.float32)
        noisy = out.astype(np.float32) + noise
        out = np.clip(noisy, 0, 255).astype(np.uint8)

    return out


# Defined A/B Test Configurations
AB_CONFIGS = [
    {
        "id": "A0_raw",
        "name": "Raw Upscale (Không can thiệp)",
        "denoise": 0.0, "color_fix": False, "recover_detail": 0.0, "sharpen": 0.0, "film_grain": 0.0,
        "purpose": "Baseline đối chứng gốc"
    },
    {
        "id": "A1_color_fix",
        "name": "+ Wavelet Color Fix",
        "denoise": 0.0, "color_fix": True, "recover_detail": 0.0, "sharpen": 0.0, "film_grain": 0.0,
        "purpose": "Khắc phục lệch màu, giữ ánh sáng gốc"
    },
    {
        "id": "A2_denoise",
        "name": "+ Khử nén nhẹ (Denoise 0.10)",
        "denoise": 0.10, "color_fix": True, "recover_detail": 0.0, "sharpen": 0.0, "film_grain": 0.0,
        "purpose": "Loại bỏ artifact nén video 720p trước AI"
    },
    {
        "id": "A3_recover",
        "name": "+ Blend giữ nhận dạng (Blend 10%)",
        "denoise": 0.10, "color_fix": True, "recover_detail": 0.10, "sharpen": 0.0, "film_grain": 0.0,
        "purpose": "Bảo vệ đường nét và biểu cảm khuôn mặt"
    },
    {
        "id": "A4_sharpen",
        "name": "+ Làm nét viền nhẹ (Sharpen 0.15)",
        "denoise": 0.10, "color_fix": True, "recover_detail": 0.10, "sharpen": 0.15, "film_grain": 0.0,
        "purpose": "Tăng độ sắc viền mắt, mi và sợi tóc"
    },
    {
        "id": "A5_film_grain",
        "name": "+ Hạt Film Grain mịn (Grain 0.008)",
        "denoise": 0.10, "color_fix": True, "recover_detail": 0.10, "sharpen": 0.15, "film_grain": 0.008,
        "purpose": "Chống da bị bệt/sáp, tạo cảm giác phim ảnh"
    },
]


def run_ab_benchmark(
    clips_dir: Path = Path("eval/clips"),
    results_dir: Path = Path("eval/results"),
    output_dir: Path = Path("eval/ab_tests"),
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    console = Console() if RICH_AVAILABLE else None

    # Use clip_A (has 1080p ground truth and clear face features)
    raw_clip_path = clips_dir / "clip_A.mp4"
    ref_clip_path = clips_dir / "clip_A_1080p.mp4"
    
    # Try finding an upscaled video from nomosuni or baseline
    upscaled_candidate = results_dir / "nomosuni" / "clip_A.mp4"
    if not upscaled_candidate.exists():
        # Fallback to bicubic 2x if no upscaled video yet
        cap = cv2.VideoCapture(str(raw_clip_path))
        ret, frame0 = cap.read()
        cap.release()
        h, w = frame0.shape[:2]
        upscaled_base = cv2.resize(frame0, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    else:
        cap = cv2.VideoCapture(str(upscaled_candidate))
        ret, upscaled_base = cap.read()
        cap.release()

    # Read ground truth frame 0
    cap_ref = cv2.VideoCapture(str(ref_clip_path))
    ret_ref, ref_frame = cap_ref.read()
    cap_ref.release()

    # Read raw 720p frame 0
    cap_raw = cv2.VideoCapture(str(raw_clip_path))
    ret_raw, raw_frame = cap_raw.read()
    cap_raw.release()

    # Align ref size to upscaled size (1440x2560)
    out_h, out_w = upscaled_base.shape[:2]
    if ref_frame.shape[:2] != (out_h, out_w):
        ref_frame_aligned = cv2.resize(ref_frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        ref_frame_aligned = ref_frame

    # Crop box for face zoom on clip_A: (ymin=0.35, xmin=0.35, ymax=0.75, xmax=0.95)
    y1, y2 = int(0.35 * out_h), int(0.75 * out_h)
    x1, x2 = int(0.35 * out_w), int(0.95 * out_w)

    ref_crop = ref_frame_aligned[y1:y2, x1:x2]
    cv2.imwrite(str(output_dir / "ref_1080p_crop.jpg"), ref_crop)

    results = []

    print("\n[A/B Test] Đang chạy đánh giá A/B cho các biến thể tiền/hậu xử lý trên Clip A...")

    for cfg in AB_CONFIGS:
        processed = apply_pipeline_stages(
            raw_frame=raw_frame,
            upscaled_frame=upscaled_base,
            denoise=cfg["denoise"],
            color_fix=cfg["color_fix"],
            recover_detail=cfg["recover_detail"],
            sharpen=cfg["sharpen"],
            film_grain=cfg["film_grain"],
        )

        psnr_val = calculate_psnr(processed, ref_frame_aligned)
        ssim_val = calculate_ssim(processed, ref_frame_aligned)
        delta_e = calculate_color_delta_e(processed, ref_frame_aligned)
        sharpness = calculate_laplacian_sharpness(processed[y1:y2, x1:x2])

        # Save face crop image
        face_crop = processed[y1:y2, x1:x2]
        crop_path = output_dir / f"{cfg['id']}_crop.jpg"
        cv2.imwrite(str(crop_path), face_crop)

        results.append({
            "id": cfg["id"],
            "name": cfg["name"],
            "purpose": cfg["purpose"],
            "psnr": round(psnr_val, 2),
            "ssim": round(ssim_val, 4),
            "delta_e": round(delta_e, 2),
            "sharpness": round(sharpness, 1),
            "crop_path": str(crop_path),
        })

    # Render side-by-side comparison strip of all A/B crops
    crop_tiles = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r in results:
        tile = cv2.imread(r["crop_path"])
        # Add label
        th, tw = tile.shape[:2]
        banner = np.zeros((45, tw, 3), dtype=np.uint8)
        tile[:45, :] = cv2.addWeighted(tile[:45, :], 0.3, banner, 0.7, 0)
        cv2.putText(tile, r["id"], (8, 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, f"SSIM:{r['ssim']} Sh:{r['sharpness']}", (8, 38), font, 0.45, (180, 240, 255), 1, cv2.LINE_AA)
        crop_tiles.append(tile)

    combined_strip = np.concatenate(crop_tiles, axis=1)
    strip_path = output_dir / "ab_comparison_strip.jpg"
    cv2.imwrite(str(strip_path), combined_strip)
    print(f"-> Đã xuất dải ảnh so sánh A/B: {strip_path}")

    # Display results
    if console:
        table = Table(title="Kết Quả Thử Nghiệm A/B: Tiền Xử Lý & Hậu Xử Lý (Clip A)")
        table.add_column("Cấu hình", style="cyan", no_wrap=True)
        table.add_column("PSNR vs 1080p (dB)", justify="right", style="magenta")
        table.add_column("SSIM", justify="right", style="blue")
        table.add_column("Delta-E (Lệch màu)", justify="right", style="yellow")
        table.add_column("Độ sắc nét (Laplacian)", justify="right", style="green")
        table.add_column("Mục tiêu kỹ thuật", style="white")

        for r in results:
            table.add_row(
                r["name"],
                str(r["psnr"]),
                str(r["ssim"]),
                str(r["delta_e"]),
                str(r["sharpness"]),
                r["purpose"],
            )
        console.print(table)
    else:
        for r in results:
            print(f"{r['name']}: PSNR={r['psnr']}, SSIM={r['ssim']}, DeltaE={r['delta_e']}, Sharpness={r['sharpness']}")

    return results


def main():
    run_ab_benchmark()


if __name__ == "__main__":
    main()
