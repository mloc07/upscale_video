"""Video Super-Resolution Evaluation and Comparison Tool.

Generates side-by-side comparison videos (face-zoom and full-frame) and compiles
benchmark metrics (time/sec, peak VRAM, PSNR, SSIM) across candidate models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ecoup.video_io import VideoReader, VideoWriter, get_video_metadata

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Pre-calibrated face crop coordinates (ymin, xmin, ymax, xmax) normalized to [0, 1]
DEFAULT_FACE_CROPS: Dict[str, Tuple[float, float, float, float]] = {
    "clip_a": (0.35, 0.35, 0.75, 0.95),  # Close-up face tilted, eyes & lips
    "clip_b": (0.15, 0.30, 0.45, 0.70),  # Upper body dancing, head & hair
    "clip_c": (0.20, 0.20, 0.65, 0.80),  # Night scene face & facial features
}
FALLBACK_CROP = (0.15, 0.25, 0.55, 0.75)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM) on luminance channel."""
    y1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(y1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(y2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(y1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(y2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(y1 * y2, (11, 11), 1.5) - mu1_mu2
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.clip(ssim_map.mean(), -1.0, 1.0))


def draw_label_banner(
    frame: np.ndarray,
    title: str,
    subtitle: Optional[str] = None,
    banner_height: int = 56,
) -> np.ndarray:
    """Draw a dark semi-transparent banner with title and subtitle at top."""
    out = frame.copy()
    h, w = out.shape[:2]
    actual_banner_h = min(banner_height, h // 5)

    # Black banner
    banner = np.zeros((actual_banner_h, w, 3), dtype=np.uint8)
    out[0:actual_banner_h, 0:w] = cv2.addWeighted(out[0:actual_banner_h, 0:w], 0.25, banner, 0.75, 0)

    # Text rendering
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_title = max(0.55, w / 900.0)
    thickness = max(1, int(round(font_scale_title * 2)))

    cv2.putText(
        out,
        title,
        (12, int(actual_banner_h * 0.48)),
        font,
        font_scale_title,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    if subtitle:
        font_scale_sub = max(0.42, w / 1200.0)
        cv2.putText(
            out,
            subtitle,
            (12, int(actual_banner_h * 0.88)),
            font,
            font_scale_sub,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def find_clip_file(directory: Path, clip_stem: str) -> Optional[Path]:
    """Find video file matching clip stem inside directory."""
    for pattern in (f"{clip_stem}.mp4", f"{clip_stem}_upscaled.mp4", f"*{clip_stem}*.mp4"):
        matches = list(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_metrics(result_dir: Path, clip_stem: str) -> Dict[str, Any]:
    """Load benchmark metrics if saved in result directory."""
    candidates = [
        result_dir / f"metrics_{clip_stem}.json",
        result_dir / "metrics.json",
        result_dir / "benchmark.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        if clip_stem in data:
                            return data[clip_stem]
                        return data
            except Exception:
                pass
    return {}


def create_comparison_video(
    original_clip_path: Path,
    model_entries: List[Tuple[str, Path, Dict[str, Any]]],
    output_path: Path,
    mode: str = "face_zoom",
    crop_box: Optional[Tuple[float, float, float, float]] = None,
    tile_size: int = 540,
) -> None:
    """Create a side-by-side synchronized comparison video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    orig_meta = get_video_metadata(original_clip_path)
    total_frames = orig_meta.total_frames
    fps = orig_meta.fps

    readers = [VideoReader(original_clip_path)]
    labels = ["Original 720p (Bicubic 2x)"]
    subtitles = [f"{orig_meta.width}x{orig_meta.height} -> 2x"]

    for model_name, path, metrics in model_entries:
        readers.append(VideoReader(path))
        labels.append(model_name)
        sub = []
        if "sec_per_sec" in metrics:
            sub.append(f"{metrics['sec_per_sec']:.2f} s/s")
        elif "fps" in metrics:
            sub.append(f"{metrics['fps']:.1f} fps")
        if "peak_vram_mb" in metrics:
            sub.append(f"{metrics['peak_vram_mb'] / 1024:.1f} GB")
        subtitles.append(" | ".join(sub) if sub else "AI Upscale")

    # Start all readers
    for r in readers:
        r.start()

    num_panels = len(readers)
    if mode == "face_zoom":
        panel_w = tile_size
        panel_h = tile_size
    else:
        # Full side-by-side: scale vertical clips so height is 960
        panel_h = 960
        aspect = orig_meta.width / orig_meta.height
        panel_w = int(round(panel_h * aspect))

    out_width = panel_w * num_panels
    out_height = panel_h

    writer = VideoWriter(
        output_path=output_path,
        width=out_width,
        height=out_height,
        fps=fps,
        audio_source_path=original_clip_path if orig_meta.has_audio else None,
        codec="auto",
    )
    writer.start()

    try:
        frame_idx = 0
        while frame_idx < total_frames:
            panel_frames = []
            for i, r in enumerate(readers):
                frame = r.read_frame()
                if frame is None:
                    break

                if i == 0:
                    # Bicubic upscale original 2x for fair comparison
                    oh, ow = frame.shape[:2]
                    frame = cv2.resize(frame, (ow * 2, oh * 2), interpolation=cv2.INTER_CUBIC)

                fh, fw = frame.shape[:2]

                if mode == "face_zoom" and crop_box:
                    ymin, xmin, ymax, xmax = crop_box
                    y1 = max(0, min(fh - 1, int(ymin * fh)))
                    y2 = max(y1 + 1, min(fh, int(ymax * fh)))
                    x1 = max(0, min(fw - 1, int(xmin * fw)))
                    x2 = max(x1 + 1, min(fw, int(xmax * fw)))
                    cropped = frame[y1:y2, x1:x2]
                    resized = cv2.resize(cropped, (panel_w, panel_h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    resized = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

                labeled = draw_label_banner(resized, labels[i], subtitles[i])
                panel_frames.append(labeled)

            if len(panel_frames) != num_panels:
                break

            combined = np.concatenate(panel_frames, axis=1)
            writer.write_frame(combined)
            frame_idx += 1

    finally:
        for r in readers:
            r.close()
        writer.close()


def run_evaluation(
    clips_dir: Path,
    results_dir: Path,
    output_dir: Path,
    models: Optional[List[str]] = None,
    mode: str = "both",
    report_path: Optional[Path] = None,
) -> None:
    """Run full evaluation and side-by-side generation."""
    console = Console() if RICH_AVAILABLE else None

    # Discover candidate models in results_dir
    available_models = [d.name for d in results_dir.iterdir() if d.is_dir()]
    if models:
        selected_models = [m for m in models if m in available_models]
    else:
        selected_models = available_models

    if not selected_models:
        msg = f"Không tìm thấy kết quả model nào trong {results_dir}."
        if console:
            console.print(f"[bold red]{msg}[/bold red]")
        else:
            print(msg)
        return

    # Find test clips (clip_A, clip_B, clip_C)
    clip_files = sorted(clips_dir.glob("clip_*.mp4"))
    # Filter out 1080p ground truth references from the main clip list
    main_clips = [c for c in clip_files if not c.stem.endswith("_1080p")]

    if not main_clips:
        print(f"Không tìm thấy clip đánh giá nào trong {clips_dir}.")
        return

    records: List[Dict[str, Any]] = []

    for clip_path in main_clips:
        stem = clip_path.stem  # e.g., clip_A
        clip_key = stem.lower()
        crop_box = DEFAULT_FACE_CROPS.get(clip_key, FALLBACK_CROP)

        # Check for 1080p ground truth reference
        ref_path = clips_dir / f"{stem}_1080p.mp4"
        ref_available = ref_path.exists()

        model_entries: List[Tuple[str, Path, Dict[str, Any]]] = []

        for model_name in selected_models:
            model_dir = results_dir / model_name
            found_vid = find_clip_file(model_dir, stem)
            if not found_vid:
                continue

            metrics = load_metrics(model_dir, stem)

            # Compute PSNR/SSIM if ground-truth 1080p reference is present
            psnr_val: Optional[float] = None
            ssim_val: Optional[float] = None
            if ref_available:
                try:
                    with VideoReader(ref_path) as ref_r, VideoReader(found_vid) as mod_r:
                        ref_r.start()
                        mod_r.start()
                        psnr_list, ssim_list = [], []
                        # Sample up to 30 frames evenly
                        step = max(1, ref_r.metadata.total_frames // 30)
                        for idx in range(ref_r.metadata.total_frames):
                            rf = ref_r.read_frame()
                            mf = mod_r.read_frame()
                            if rf is None or mf is None:
                                break
                            if idx % step == 0:
                                # Align dimensions to reference
                                if mf.shape[:2] != rf.shape[:2]:
                                    mf = cv2.resize(mf, (rf.shape[1], rf.shape[0]), interpolation=cv2.INTER_AREA)
                                psnr_list.append(calculate_psnr(mf, rf))
                                ssim_list.append(calculate_ssim(mf, rf))
                        if psnr_list:
                            psnr_val = float(np.mean(psnr_list))
                            ssim_val = float(np.mean(ssim_list))
                except Exception as e:
                    print(f"Lỗi tính PSNR/SSIM cho {model_name} on {stem}: {e}")

            model_entries.append((model_name, found_vid, metrics))

            records.append({
                "clip": stem,
                "model": model_name,
                "path": str(found_vid),
                "sec_per_sec": metrics.get("sec_per_sec", metrics.get("processing_time_sec", 0.0) / 6.0 if metrics.get("processing_time_sec") else "-"),
                "vram_mb": metrics.get("peak_vram_mb", "-"),
                "psnr": f"{psnr_val:.2f}" if psnr_val is not None else "-",
                "ssim": f"{ssim_val:.4f}" if ssim_val is not None else "-",
            })

        if not model_entries:
            print(f"[{stem}] Chưa có kết quả model nào, bỏ qua render.")
            continue

        print(f"\n[{stem}] Đang kết xuất video so sánh cho {len(model_entries)} model...")

        modes_to_run = ["face_zoom", "full"] if mode == "both" else [mode]
        for m in modes_to_run:
            out_name = f"{stem}_{m}.mp4"
            out_file = output_dir / out_name
            print(f"  -> Kết xuất: {out_file} (chế độ {m})...")
            create_comparison_video(
                original_clip_path=clip_path,
                model_entries=model_entries,
                output_path=out_file,
                mode=m,
                crop_box=crop_box,
            )
            print(f"  -> Hoàn tất {out_file}!")

    # Display benchmark table
    if console and records:
        table = Table(title="Bảng Tổng Hợp Đánh Giá Video Super-Resolution")
        table.add_column("Clip", style="cyan", no_wrap=True)
        table.add_column("Mô hình", style="green", no_wrap=True)
        table.add_column("Thời gian (s/giây video)", justify="right")
        table.add_column("VRAM đỉnh (MB)", justify="right", style="yellow")
        table.add_column("PSNR (dB)", justify="right", style="magenta")
        table.add_column("SSIM", justify="right", style="blue")

        for r in records:
            table.add_row(
                r["clip"],
                r["model"],
                str(r["sec_per_sec"]),
                str(r["vram_mb"]),
                r["psnr"],
                r["ssim"],
            )
        console.print(table)
    elif records:
        print("\n=== BẢNG TỔNG HỢP ĐÁNH GIÁ ===")
        for r in records:
            print(f"{r['clip']} | {r['model']} | s/s: {r['sec_per_sec']} | VRAM: {r['vram_mb']}MB | PSNR: {r['psnr']} | SSIM: {r['ssim']}")

    # Export report if requested
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_out_dir = str(output_dir.resolve()).replace("\\", "/")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# BÁO CÁO ĐÁNH GIÁ & SO SÁNH CÁC MÔ HÌNH VIDEO SUPER-RESOLUTION\n\n")
            f.write("## 1. Bảng số đo định lượng\n\n")
            f.write("| Clip | Mô hình | Thời gian (s/giây video) | VRAM đỉnh (MB) | PSNR vs 1080p (dB) | SSIM |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in records:
                f.write(f"| {r['clip']} | {r['model']} | {r['sec_per_sec']} | {r['vram_mb']} | {r['psnr']} | {r['ssim']} |\n")
            f.write("\n## 2. Video so sánh trực quan đã kết xuất\n\n")
            for clip_path in main_clips:
                stem = clip_path.stem
                f.write(f"### {stem}\n")
                f.write(f"- [Video soi chi tiết mặt (Face Zoom)](file:///{resolved_out_dir}/{stem}_face_zoom.mp4)\n")
                f.write(f"- [Video toàn khung cạnh nhau (Full Frame)](file:///{resolved_out_dir}/{stem}_full.mp4)\n\n")
        print(f"\nĐã lưu báo cáo đánh giá vào: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="So sánh và đánh giá các mô hình Video Super-Resolution.")
    parser.add_argument("--clips-dir", type=Path, default=Path("eval/clips"), help="Thư mục chứa clip gốc 720p")
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"), help="Thư mục chứa kết quả của các model")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/comparisons"), help="Thư mục xuất video so sánh")
    parser.add_argument("--models", nargs="+", default=None, help="Danh sách tên model cần so sánh")
    parser.add_argument("--mode", choices=["both", "face_zoom", "full"], default="both", help="Chế độ kết xuất video")
    parser.add_argument("--report-path", type=Path, default=Path("docs/eval_report.md"), help="Đường dẫn file báo cáo markdown")
    args = parser.parse_args()

    run_evaluation(
        clips_dir=args.clips_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        models=args.models,
        mode=args.mode,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
