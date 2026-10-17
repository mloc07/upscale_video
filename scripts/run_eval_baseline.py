"""Script to run baseline or candidate model inference on eval/clips.

Saves upscaled videos to eval/results/<model_name>/ and records time and VRAM metrics.
Works seamlessly on both local environments (ONNX) and Google Colab (CUDA / Diffusion).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Avoid console encoding issues on Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ecoup.config import (
    AppConfig,
    CheckpointConfig,
    EncodeConfig,
    FrameSavingConfig,
    InferenceConfig,
    ModelConfig,
    ThermalConfig,
)
from ecoup.pipeline import UpscalePipeline
from ecoup.utils import get_gpu_vram_info
from ecoup.video_io import get_video_metadata


def get_peak_vram_mb() -> float:
    """Query current peak VRAM usage via PyTorch or pynvml."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except Exception:
        pass

    try:
        info = get_gpu_vram_info()
        if info:
            return round(info.used_mb, 1)
    except Exception:
        pass

    return 0.0


def run_clip_evaluation(
    model_name: str,
    clips_dir: Path,
    output_dir: Path,
    tile_size: int = 0,
    control_strength: float = 0.75,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Engine
    engine = None
    target_scale = 2
    model_scale = 2

    if model_name.lower() == "ccsr":
        from ecoup.engine.ccsr_engine import CcsrUpscaleEngine
        print("[Eval] Đang khởi tạo CCSR Diffusion Engine...")
        engine = CcsrUpscaleEngine(control_strength=control_strength)
        engine.load(model_path="preset/models", precision="fp16", provider="cuda")
        model_scale = engine.scale
    elif model_name.lower() == "nomosuni":
        from ecoup.engine.onnx_engine import OnnxUpscaleEngine
        model_path = Path("models/2xNomosUni_compact_otf_medium.onnx")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"[Eval] Đang nạp ONNX NomosUni từ {model_path}...")
        engine = OnnxUpscaleEngine()
        engine.load(model_path, provider="auto")
        model_scale = 2
    elif model_name.lower() == "realesrgan":
        from ecoup.engine.onnx_engine import OnnxUpscaleEngine
        model_path = Path("models/2x-realesrgan-x2plus.onnx")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"[Eval] Đang nạp ONNX RealESRGAN từ {model_path}...")
        engine = OnnxUpscaleEngine()
        engine.load(model_path, provider="auto")
        model_scale = 2
    else:
        raise ValueError(f"Model không xác định: {model_name}. Hỗ trợ: ccsr, nomosuni, realesrgan")

    # Find test clips (excluding _1080p ground truth references)
    clip_files = sorted(clips_dir.glob("clip_*.mp4"))
    main_clips = [c for c in clip_files if not c.stem.endswith("_1080p")]

    if not main_clips:
        print(f"Không tìm thấy clip nào trong {clips_dir}")
        return

    print(f"\n[Eval] Bắt đầu đánh giá mô hình: {model_name} trên {len(main_clips)} clips...")

    for clip_path in main_clips:
        stem = clip_path.stem
        out_video = output_dir / f"{stem}.mp4"
        meta = get_video_metadata(clip_path)

        print(f"\n{'=' * 60}")
        print(f"Đang xử lý: {stem} ({meta.width}x{meta.height}, {meta.total_frames} frames, {meta.duration_seconds:.2f}s)")
        print(f"{'=' * 60}")

        cfg = AppConfig(
            model=ModelConfig(
                name=model_name,
                path="preset/models/controlnet" if model_name == "ccsr" else str(model_path),
                scale=model_scale,
                target_scale=target_scale,
            ),
            inference=InferenceConfig(
                provider="auto",
                tile_size=tile_size,
                tile_overlap=16 if tile_size > 0 else 0,
                control_strength=control_strength,
            ),
            thermal=ThermalConfig(enabled=False),
            frame_saving=FrameSavingConfig(enable_dedup=False),
            checkpoint=CheckpointConfig(enabled=False),
            encode=EncodeConfig(codec="auto", copy_audio=True),
        )

        pipeline = UpscalePipeline(config=cfg, engine=engine)

        # Reset peak memory tracker if PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        t0 = time.perf_counter()
        pipeline.run(clip_path, out_video, resume=False)
        elapsed = time.perf_counter() - t0

        peak_vram = get_peak_vram_mb()
        sec_per_sec = round(elapsed / meta.duration_seconds, 2) if meta.duration_seconds > 0 else 0.0
        fps = round(meta.total_frames / elapsed, 2) if elapsed > 0 else 0.0

        metrics = {
            "clip_name": stem,
            "model_name": model_name,
            "input_resolution": f"{meta.width}x{meta.height}",
            "output_resolution": f"{meta.width * 2}x{meta.height * 2}",
            "total_frames": meta.total_frames,
            "video_duration_sec": meta.duration_seconds,
            "processing_time_sec": round(elapsed, 2),
            "sec_per_sec": sec_per_sec,
            "fps": fps,
            "peak_vram_mb": peak_vram,
        }

        metrics_file = output_dir / f"metrics_{stem}.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"-> Hoàn tất {stem} trong {elapsed:.2f}s ({sec_per_sec} s/s, {fps} fps, peak VRAM: {peak_vram} MB)")

    print(f"\n[Eval] Hoàn tất toàn bộ bộ test cho model: {model_name}!")


def main():
    parser = argparse.ArgumentParser(description="Chạy suy luận benchmark trên các clip đánh giá.")
    parser.add_argument("--model", type=str, required=True, choices=["ccsr", "nomosuni", "realesrgan"], help="Tên model")
    parser.add_argument("--clips-dir", type=Path, default=Path("eval/clips"), help="Thư mục clip đầu vào")
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"), help="Thư mục lưu kết quả")
    parser.add_argument("--tile-size", type=int, default=0, help="Kích thước tile (0 = toàn khung)")
    parser.add_argument("--control-strength", type=float, default=0.75, help="Cường độ CCSR control strength")
    args = parser.parse_args()

    out_dir = args.results_dir / args.model
    run_clip_evaluation(
        model_name=args.model,
        clips_dir=args.clips_dir,
        output_dir=out_dir,
        tile_size=args.tile_size,
        control_strength=args.control_strength,
    )


if __name__ == "__main__":
    main()
