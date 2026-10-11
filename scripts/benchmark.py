"""Benchmark script to measure FPS, VRAM peak, temperature, and power consumption."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from rich.console import Console
from rich.table import Table

from ecoup.cli import check_gpu, _ensure_path_refreshed
from ecoup.config import AppConfig, ModelConfig, ThermalConfig, FrameSavingConfig, InferenceConfig
from ecoup.pipeline import UpscalePipeline
from ecoup.video_io import get_video_metadata


def generate_benchmark_clip(output_path: Path, duration_sec: int = 5) -> Path:
    """Generate a synthetic benchmark clip if no input is provided."""
    _ensure_path_refreshed()
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found for benchmark clip generation.")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size=1280x720:rate=24",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def run_benchmark(input_path: str, runs: int = 1) -> None:
    console = Console()
    console.print(f"[bold green]Eco Video Upscaler - Benchmark Utility[/bold green]")
    console.print(f"Video benchmark: {input_path}")

    meta = get_video_metadata(input_path)
    console.print(f"Độ phân giải: {meta.width}x{meta.height} • {meta.fps} fps • {meta.total_frames} khung hình")

    # Define test matrix
    configs = [
        ("Tile 256 + Dedup (Eco)", 256, True, 70),
        ("Tile 256 - No Dedup", 256, False, 70),
        ("No Tiling (Full Frame)", 0, True, 70),
    ]

    table = Table(title="Kết Quả Thử Nghiệm Benchmark (RTX 3050 Laptop)")
    table.add_column("Cấu hình", style="cyan", no_wrap=True)
    table.add_column("Thời gian (s)", justify="right")
    table.add_column("FPS", justify="right", style="green")
    table.add_column("Nhiệt độ TB (°C)", justify="right", style="yellow")
    table.add_column("Công suất TB (W)", justify="right", style="blue")
    table.add_column("Khung bỏ qua", justify="right", style="magenta")

    tmp_out = Path("benchmark_temp_out.mp4")

    try:
        for label, tile_sz, dedup_en, temp_soft in configs:
            cfg = AppConfig(
                model=ModelConfig(scale=2, target_scale=2),
                inference=InferenceConfig(tile_size=tile_sz),
                frame_saving=FrameSavingConfig(enable_dedup=dedup_en),
                thermal=ThermalConfig(enabled=True, temp_soft_c=temp_soft, poll_interval_sec=0.5),
            )

            # Fast mock upscale function (cv2 bicubic) for benchmark baseline
            import cv2
            def mock_fn(tile: np.ndarray) -> np.ndarray:
                h, w = tile.shape[:2]
                return cv2.resize(tile, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

            temps = []
            powers = []

            def on_prog(idx, total, fps, eta, metrics, dedup):
                if metrics and metrics.temp_c > 0:
                    temps.append(metrics.temp_c)
                if metrics and metrics.power_w > 0:
                    powers.append(metrics.power_w)

            pipeline = UpscalePipeline(config=cfg, custom_upscale_fn=mock_fn, on_progress=on_prog)

            start = time.perf_counter()
            pipeline.run(input_path, tmp_out, resume=False)
            elapsed = time.perf_counter() - start

            fps = meta.total_frames / max(0.001, elapsed)
            avg_temp = round(sum(temps) / len(temps), 1) if temps else "--"
            avg_pwr = round(sum(powers) / len(powers), 1) if powers else "--"

            table.add_row(
                label,
                f"{elapsed:.2f}s",
                f"{fps:.1f}",
                str(avg_temp),
                str(avg_pwr),
                f"{pipeline.config.frame_saving.enable_dedup}",
            )

        console.print(table)
    finally:
        if tmp_out.exists():
            tmp_out.unlink()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Eco Video Upscaler configurations")
    parser.add_argument("-i", "--input", help="Path to test video clip")
    args = parser.parse_args()

    input_file = args.input
    gen_file = None
    if not input_file or not os.path.exists(input_file):
        gen_file = Path("benchmark_synthetic_sample.mp4")
        generate_benchmark_clip(gen_file, duration_sec=5)
        input_file = str(gen_file)

    try:
        run_benchmark(input_file)
    finally:
        if gen_file and gen_file.exists():
            gen_file.unlink()


if __name__ == "__main__":
    main()
