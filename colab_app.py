"""Gradio Web Application for Eco Video Upscaler (Optimized for Google Colab and remote devices).

Utilitarian, minimalist interface designed for Phones and Laptops without video previews:
1. Input File Picker
2. Configuration Controls (Model, Tile, Filters)
3. Action Buttons: Run & Cancel
4. Progress Bar & Realtime Hardware/VRAM Telemetry
5. Output Download Button
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Avoid UnicodeEncodeError on Windows consoles (cp1252) when printing Vietnamese text
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ecoup.config import (
    AppConfig,
    ModelConfig,
    ThermalConfig,
    FrameSavingConfig,
    CheckpointConfig,
    EncodeConfig,
    InferenceConfig,
    PreProcessConfig,
    PostProcessConfig,
)
from ecoup.engine.onnx_engine import OnnxUpscaleEngine
from ecoup.pipeline import UpscalePipeline
from ecoup.utils import get_gpu_vram_info
from ecoup.video_io import get_video_metadata

# Global tracker for cancel functionality
active_pipeline: Optional[UpscalePipeline] = None

# Verified models mapping: model_key -> (filename, download_url, default_scale)
MODEL_REGISTRY = {
    "nomosuni": (
        "2xNomosUni_compact_otf_medium.onnx",
        "https://huggingface.co/notaneimu/onnx-image-models/resolve/main/2xNomosUni_compact_otf_medium.onnx",
        2,
    ),
    "x2plus": (
        "2x-realesrgan-x2plus.onnx",
        "https://huggingface.co/notaneimu/onnx-image-models/resolve/main/2x-realesrgan-x2plus.onnx",
        2,
    ),
    "general": (
        "realesr-general-x4v3.onnx",
        "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-general-x4v3.onnx",
        4,
    ),
    "anime": (
        "realesr-animevideov3.onnx",
        "https://huggingface.co/nimjinwei/designerplatform-models/resolve/main/realesr-animevideov3.onnx",
        4,
    ),
    "span": (
        "2x-spanx2-ch48.onnx",
        "https://huggingface.co/notaneimu/onnx-image-models/resolve/main/2x-spanx2-ch48.onnx",
        2,
    ),
}


def get_available_models():
    return [
        "NomosUni 2x (Mặc định - Nét nhất, chi tiết chân thực 2.3MB)",
        "RealESRGAN x2plus (Chi tiết cao cấp 67MB - Khuyên dùng trên Cloud)",
        "realesr-general-x4v3 (4x, Khử nhiễu mịn da 4.9MB)",
        "realesr-animevideov3 (4x, Hoạt hình / 2D / Anime 2.5MB)",
        "SPAN 2x (Siêu nhanh, mát GPU 1.6MB)",
    ]


def ensure_model_file(model_key: str) -> tuple[Path, int]:
    """Ensure ONNX model file exists on disk, auto-downloading if necessary."""
    if model_key not in MODEL_REGISTRY:
        model_key = "nomosuni"
    filename, url, scale = MODEL_REGISTRY[model_key]
    dest = Path("models") / filename
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Colab App] Model chưa có sẵn, đang tải tự động từ: {url}")
        urllib.request.urlretrieve(url, str(dest))
        print(f"[Colab App] Tải model thành công: {dest}")
    return dest, scale


def cancel_upscale():
    """Cancel currently running upscale job."""
    global active_pipeline
    if active_pipeline is not None:
        active_pipeline.cancel()
        return "⏹️ Đã gửi lệnh hủy! Đang dừng an toàn và giải phóng GPU..."
    return "Không có tiến trình nào đang chạy."


def run_upscale(
    input_file_path: Optional[str],
    model_choice: str,
    target_scale_choice: str,
    tile_choice: str,
    pre_denoise_pct: int,
    recover_detail_pct: int,
    sharpen_pct: int,
    film_grain_checked: bool,
    progress=gr.Progress(track_tqdm=False),
):
    """Execute video upscaling pipeline with real-time status and cancellation support."""
    global active_pipeline

    if not input_file_path or not os.path.exists(input_file_path):
        raise gr.Error("Vui lòng chọn video đầu vào hợp lệ.")

    in_path = Path(input_file_path)
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{in_path.stem}_upscaled.mp4"

    # Scale mapping
    target_scale = 4 if "4x" in target_scale_choice else 2

    # Tile mapping:
    # On Cloud (Colab T4 15GB VRAM), full-frame (tile_sz=0) processes the entire frame at once,
    # utilizing 1.5 - 3.5GB VRAM and running 3x-5x faster.
    if "toàn khung" in tile_choice.lower() or "không chia" in tile_choice.lower():
        tile_sz = 0
        tile_ov = 0
    elif "256" in tile_choice:
        tile_sz = 256
        tile_ov = 16
    else:  # 512x512
        tile_sz = 512
        tile_ov = 16

    progress(0.01, desc="Đang chuẩn bị mô hình AI...")

    # Model resolution
    engine = None
    custom_fn = None
    model_name = "nomosuni"
    model_scale = 2

    if "ccsr" in model_choice.lower():
        # Check if CCSR weights and dependencies are available
        ccsr_weights = Path("models/real-world_ccsr-fp16.safetensors")
        ccsr_v2_pth = Path("models/ccsr_v2.pth")
        if ccsr_weights.exists() or ccsr_v2_pth.exists():
            try:
                from ecoup.engine.ccsr_engine import CcsrUpscaleEngine
                engine = CcsrUpscaleEngine()
                engine.load(model_path=str(ccsr_v2_pth if ccsr_v2_pth.exists() else ccsr_weights), provider="cuda")
                model_scale = 2
                model_name = "ccsr_v2"
            except Exception as e:
                print(f"[Colab App] CCSR engine error: {e}. Sử dụng RealESRGAN x2plus sắc nét.")
                mf, model_scale = ensure_model_file("x2plus")
                model_name = "RealESRGAN_x2plus"
                engine = OnnxUpscaleEngine()
                engine.load(mf, provider="auto")
        else:
            print("[Colab App] Trọng số CCSR (~3.5GB) chưa có trong models/. Tự động dùng RealESRGAN x2plus (67MB).")
            mf, model_scale = ensure_model_file("x2plus")
            model_name = "RealESRGAN_x2plus"
            engine = OnnxUpscaleEngine()
            engine.load(mf, provider="auto")
    elif "x2plus" in model_choice.lower():
        mf, model_scale = ensure_model_file("x2plus")
        model_name = "RealESRGAN_x2plus"
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    elif "anime" in model_choice.lower():
        mf, model_scale = ensure_model_file("anime")
        model_name = "realesr-animevideov3"
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    elif "span" in model_choice.lower():
        mf, model_scale = ensure_model_file("span")
        model_name = "SPAN_2x"
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    elif "general" in model_choice.lower():
        mf, model_scale = ensure_model_file("general")
        model_name = "realesr-general-x4v3"
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    else:
        # Default NomosUni 2x
        mf, model_scale = ensure_model_file("nomosuni")
        model_name = "NomosUni_2x"
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")

    if engine is not None and hasattr(engine, "scale") and engine.scale > 0:
        model_scale = engine.scale

    cfg = AppConfig(
        model=ModelConfig(
            name=model_name,
            path=str(mf) if "mf" in locals() else "models/model.onnx",
            scale=model_scale,
            target_scale=target_scale,
        ),
        inference=InferenceConfig(
            provider="auto",
            tile_size=tile_sz,
            tile_overlap=tile_ov,
        ),
        thermal=ThermalConfig(enabled=False),  # Cloud GPU runs at full power
        frame_saving=FrameSavingConfig(enable_dedup=True),
        checkpoint=CheckpointConfig(enabled=False),
        encode=EncodeConfig(codec="auto", copy_audio=True),
        pre_process=PreProcessConfig(denoise=pre_denoise_pct / 100.0),
        post_process=PostProcessConfig(
            recover_detail=recover_detail_pct / 100.0,
            sharpen=sharpen_pct / 100.0,
            film_grain=0.015 if film_grain_checked else 0.0,
        ),
    )

    last_status = "Đang bắt đầu xử lý..."

    def on_progress(frame_idx, total_frames, fps, eta_sec, metrics, dedup_rate):
        nonlocal last_status
        pct = (frame_idx / max(1, total_frames))
        mins = int(eta_sec // 60)
        secs = int(eta_sec % 60)

        # Query hardware GPU VRAM metrics
        used_gb, total_gb, gpu_name = get_gpu_vram_info()
        vram_info = ""
        if total_gb > 0:
            vram_pct = (used_gb / total_gb) * 100
            vram_info = f" • VRAM: {used_gb:.1f}/{total_gb:.1f}GB ({vram_pct:.0f}%)"

        prov = getattr(engine, "active_provider", "GPU")
        if "cuda" in prov.lower():
            prov_label = "CUDA (NVIDIA T4)"
        elif "dml" in prov.lower():
            prov_label = "DirectML"
        else:
            prov_label = prov

        last_status = (
            f"Khung hình {frame_idx}/{total_frames} ({int(pct * 100)}%) "
            f"• {fps:.1f} fps • Còn lại: {mins:02d}:{secs:02d}{vram_info} • GPU: {prov_label}"
        )
        progress(pct, desc=last_status)

    pipeline = UpscalePipeline(
        config=cfg,
        engine=engine,
        custom_upscale_fn=custom_fn,
        on_progress=on_progress,
    )
    active_pipeline = pipeline

    try:
        progress(0.02, desc="Đang phân tích video và khởi động GPU...")
        pipeline.run(in_path, out_path, resume=False)
    except Exception as e:
        if pipeline._cancelled:
            return None, "⏹️ Quá trình xử lý đã được người dùng HỦY BỎ thành công."
        raise e
    finally:
        active_pipeline = None

    if pipeline._cancelled:
        return None, "⏹️ Quá trình xử lý đã được người dùng HỦY BỎ thành công."

    progress(1.0, desc="Hoàn tất upscale! Video đã sẵn sàng tải về.")
    return str(out_path), f"Hoàn tất thành công! Video đã lưu: {out_path.name}"


def create_ui():
    """Create minimalist, utilitarian Gradio interface."""
    custom_css = """
    #title-box { text-align: center; margin-bottom: 8px; }
    #title-box h1 { font-size: 1.5rem; margin-bottom: 4px; }
    #title-box p { color: #888; font-size: 0.9rem; }
    .gr-button-primary { background: #2563eb !important; }
    .gr-button-stop { background: #dc2626 !important; color: white !important; }
    """

    with gr.Blocks(title="Eco Video Upscaler (Cloud Edition)", theme=gr.themes.Base(), css=custom_css) as demo:
        with gr.Column(elem_id="title-box"):
            gr.Markdown(
                """
                # ⚡ Eco Video Upscaler (Cloud Edition)
                Giao diện tinh gọn phục chế & siêu phân giải video trên Cloud GPU — Không tốn tài nguyên máy tính cá nhân.
                """
            )

        with gr.Group():
            gr.Markdown("### 1. Đầu vào:")
            in_file = gr.File(
                label="Chọn video từ thiết bị (Điện thoại hoặc Laptop)",
                file_types=[".mp4", ".mov", ".mkv", ".avi", ".webm"],
                file_count="single",
                type="filepath",
            )

        with gr.Group():
            gr.Markdown("### 2. Cài đặt:")
            with gr.Row():
                model_dd = gr.Dropdown(
                    choices=get_available_models(),
                    value="NomosUni 2x (Mặc định - Nét nhất, chi tiết chân thực 2.3MB)",
                    label="Mô hình AI",
                    info="Chọn RealESRGAN x2plus để tận dụng tối đa GPU Cloud, hoặc NomosUni 2x để nét tự nhiên.",
                )
                scale_dd = gr.Dropdown(
                    choices=["1080p (Scale 2x)", "1440p (Scale 2x)", "4K (Scale 4x)"],
                    value="1080p (Scale 2x)",
                    label="Độ phân giải đích",
                )
                tile_dd = gr.Dropdown(
                    choices=[
                        "Toàn khung (Nhanh nhất - Tận dụng tối đa GPU Cloud 15GB VRAM)",
                        "Tile 512x512 (Cân bằng GPU & VRAM)",
                        "Tile 256x256 (Tiết kiệm VRAM - Dành cho máy yếu)",
                    ],
                    value="Toàn khung (Nhanh nhất - Tận dụng tối đa GPU Cloud 15GB VRAM)",
                    label="Chế độ xử lý (Chia Tile)",
                    info="Trên Colab T4 (15GB VRAM), chế độ 'Toàn khung' chạy nhanh gấp 3-5 lần và ăn 1.5 - 3.5GB VRAM.",
                )

            with gr.Row():
                denoise_sl = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=30,
                    step=5,
                    label="Khử nhiễu gốc - Pre-Denoise (%)",
                    info="Lọc sạch hạt rác nén trước khi AI vẽ để chống vẽ bậy",
                )
                recover_sl = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Khôi phục gốc - Recover Detail (%)",
                    info="Trộn texture video gốc để chống bệt da/môi",
                )

            with gr.Row():
                sharpen_sl = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=20,
                    step=5,
                    label="Làm nét viền - Sharpen (%)",
                    info="Tăng tương phản cạnh giúp viền rõ ràng",
                )
                grain_cb = gr.Checkbox(
                    value=True,
                    label="Bù hạt tự nhiên (Film Grain)",
                    info="Khử cảm giác da búp bê sáp / silicon",
                )

        with gr.Group():
            gr.Markdown("### 3. Thực hiện:")
            with gr.Row():
                btn_run = gr.Button("🚀 BẮT ĐẦU UPSCALE", variant="primary", size="lg", scale=3)
                btn_cancel = gr.Button("⏹️ HỦY BỎ / CANCEL", variant="stop", size="lg", scale=1)

            status_output = gr.Textbox(
                label="Trạng thái xử lý",
                value="Sẵn sàng...",
                interactive=False,
                lines=2,
            )

        with gr.Group():
            gr.Markdown("### 4. Thành phẩm:")
            out_file = gr.File(
                label="Tệp video thành phẩm (Bấm vào đây để tải về thiết bị)",
                interactive=False,
            )

        # Handle Run & Cancel
        run_event = btn_run.click(
            fn=run_upscale,
            inputs=[in_file, model_dd, scale_dd, tile_dd, denoise_sl, recover_sl, sharpen_sl, grain_cb],
            outputs=[out_file, status_output],
        )

        btn_cancel.click(
            fn=cancel_upscale,
            outputs=[status_output],
            cancels=[run_event],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Eco Video Upscaler - Gradio Cloud WebUI")
    parser.add_argument("--share", action="store_true", help="Generate public gradio.live link")
    parser.add_argument("--port", type=int, default=7860, help="Local server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    args = parser.parse_args()

    demo = create_ui()
    print("\n=======================================================")
    print("🚀 Khởi động Eco Video Upscaler Cloud WebUI...")
    print("=======================================================\n")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
