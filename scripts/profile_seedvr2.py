"""Profile SeedVR2 per-step execution time, peak VRAM, and precision on GPU.

Measures:
- Input preparation
- VAE Encode (GPU transfer + Conv3D encode + CPU offload)
- DiT Sampling (GPU transfer + DiT forward + CPU offload)
- VAE Decode (GPU transfer + Conv3D decode + CPU offload)
- Post-process resizing
- Peak VRAM allocated
- Overall throughput (FPS and ms per frame)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ecoup.engine.seedvr2_engine import SeedVR2UpscaleEngine


def read_clip(video_path: str, max_frames: int = 35) -> list[np.ndarray]:
    """Read up to max_frames RGB frames from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không thể mở video tại: {video_path}")
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Không đọc được khung hình nào từ: {video_path}")
    return frames


def run_profile(
    video_path: str,
    frames_count: int = 35,
    precision: str = "fp16",
    memory_mode: str = "auto",
    sub_len: int = 5,
    model_resolution: str = "native",
    model_path: str = "ckpts/seedvr2_ema_3b.pth",
    output_md: str | None = None,
    output_json: str | None = None,
) -> dict:
    print("=" * 60)
    print(f"Bắt đầu Profile SeedVR2-3B: precision={precision}, memory_mode={memory_mode}, sub_len={sub_len}, res={model_resolution}, frames={frames_count}")
    print(f"Video nguồn: {video_path}")
    print("=" * 60)

    try:
        import torch
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    except ImportError:
        has_cuda = False
        gpu_name = "No PyTorch"

    frames = read_clip(video_path, max_frames=frames_count)
    actual_frames = len(frames)
    h, w = frames[0].shape[:2]
    print(f"Đã đọc {actual_frames} khung hình ({w}x{h} RGB)")

    engine = SeedVR2UpscaleEngine(
        scale=2,
        precision=precision,
        memory_mode=memory_mode,
        sub_len=sub_len,
        model_resolution=model_resolution,
        enable_profiling=True,
    )
    engine.load(model_path=model_path, precision=precision, provider="cuda" if has_cuda else "cpu")

    if has_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    _ = engine.upscale_clip(frames)
    if has_cuda:
        torch.cuda.synchronize()
    total_duration = time.perf_counter() - t_start

    stats = getattr(engine, "last_profile_stats", {})
    timings = stats.get("timings", {})
    peak_vram_mb = stats.get("peak_vram_mb", 0.0)

    fps = actual_frames / max(0.001, total_duration)
    ms_per_frame = (total_duration / actual_frames) * 1000.0

    profile_summary = {
        "gpu_name": gpu_name,
        "precision": precision,
        "input_resolution": f"{w}x{h}",
        "total_frames": actual_frames,
        "total_time_sec": round(total_duration, 3),
        "fps": round(fps, 3),
        "ms_per_frame": round(ms_per_frame, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "peak_vram_gb": round(peak_vram_mb / 1024.0, 2),
        "step_timings_sec": {k: round(v, 4) for k, v in timings.items()},
    }

    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐO ĐẠC PROFILE")
    print("=" * 60)
    print(f"GPU: {gpu_name}")
    print(f"Precision: {precision}")
    print(f"Độ phân giải vào: {w}x{h} -> Ra: {w*2}x{h*2}")
    print(f"Số khung xử lý: {actual_frames}")
    print(f"Tổng thời gian: {total_duration:.2f}s ({fps:.2f} FPS, {ms_per_frame:.1f} ms/frame)")
    print(f"VRAM đỉnh: {peak_vram_mb:.1f} MB ({peak_vram_mb/1024.0:.2f} GB)")
    print("-" * 60)
    print("Chi tiết từng bước:")
    for step_name, dur in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        pct = (dur / max(0.001, total_duration)) * 100.0
        print(f"  - {step_name:<24}: {dur:6.3f}s ({pct:5.1f}%)")
    print("=" * 60)

    # Generate Markdown Table
    md_lines = [
        f"### SeedVR2 Profile: {gpu_name} ({precision}, memory_mode={engine.effective_memory_mode}, sub_len={sub_len}, res={model_resolution})",
        "",
        f"- **Độ phân giải nguồn:** {w}x{h} ({actual_frames} khung hình)",
        f"- **VRAM đỉnh:** {peak_vram_mb/1024.0:.2f} GB",
        f"- **Tốc độ:** {fps:.2f} FPS ({ms_per_frame:.1f} ms/frame, tổng {total_duration:.2f}s)",
        "",
        "| Bước xử lý | Thời gian (s) | Tỉ lệ (%) |",
        "|---|:---:|:---:|",
    ]
    for step_name, dur in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        pct = (dur / max(0.001, total_duration)) * 100.0
        md_lines.append(f"| `{step_name}` | {dur:.3f} | {pct:.1f}% |")
    md_lines.append(f"| **Tổng cộng** | **{total_duration:.3f}** | **100.0%** |")
    md_content = "\n".join(md_lines) + "\n"

    if output_md:
        out_p = Path(output_md)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "a" if out_p.exists() else "w", encoding="utf-8") as f:
            f.write("\n" + md_content)
        print(f"Đã lưu kết quả Markdown vào: {output_md}")

    if output_json:
        out_j = Path(output_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        with open(out_j, "w", encoding="utf-8") as f:
            json.dump(profile_summary, f, indent=2)
        print(f"Đã lưu kết quả JSON vào: {output_json}")

    engine.close()
    return profile_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SeedVR2 Performance Profiler")
    parser.add_argument("--video", type=str, default="plan_for_project/tests_video/720_A.mp4", help="Đường dẫn video test")
    parser.add_argument("--frames", type=int, default=10, help="Số khung hình test")
    parser.add_argument("--precision", type=str, choices=["bfloat16", "fp16"], default="fp16", help="Precision (fp16 / bfloat16)")
    parser.add_argument("--memory_mode", type=str, choices=["auto", "resident", "offload"], default="auto", help="Memory mode (auto / resident / offload)")
    parser.add_argument("--sub_len", type=int, default=5, help="Temporal window (5, 9, 13...)")
    parser.add_argument("--model_resolution", type=str, choices=["native", "match_target"], default="native", help="Model resolution")
    parser.add_argument("--ckpt", type=str, default="ckpts/seedvr2_ema_3b.pth", help="Đường dẫn file .pth")
    parser.add_argument("--output_md", type=str, default="docs/benchmark.md", help="File markdown xuất kết quả")
    parser.add_argument("--output_json", type=str, default=None, help="File JSON xuất kết quả")
    args = parser.parse_args()

    run_profile(
        video_path=args.video,
        frames_count=args.frames,
        precision=args.precision,
        memory_mode=args.memory_mode,
        sub_len=args.sub_len,
        model_resolution=args.model_resolution,
        model_path=args.ckpt,
        output_md=args.output_md,
        output_json=args.output_json,
    )
