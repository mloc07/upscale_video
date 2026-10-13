"""Gradio Web Application for Eco Video Upscaler (Optimized for Google Colab and remote devices).

Utilitarian, minimalist interface designed for Phones and Laptops without video previews:
1. Input File Picker
2. Configuration Controls
3. Progress Bar & Realtime Status
4. Output Download Button
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

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
from ecoup.video_io import get_video_metadata


def get_available_models():
    return [
        "CCSR v2 (Content-Consistent - Khuyên dùng cho Cloud)",
        "NomosUni 2x (Siêu nét, nhẹ 2.3MB)",
        "SPAN 2x (Siêu nhanh, mát GPU 1.6MB)",
        "realesr-general-x4v3 (4x, Khử nhiễu mịn da 4.9MB)",
        "realesr-animevideov3 (4x, Hoạt hình / 2D 2.5MB)",
        "RealESRGAN x2plus (2x, Chi tiết cao cấp 67MB)",
    ]


def run_upscale(
    input_file_path: Optional[str],
    model_choice: str,
    target_scale_choice: str,
    pre_denoise_pct: int,
    recover_detail_pct: int,
    sharpen_pct: int,
    film_grain_checked: bool,
    progress=gr.Progress(track_tqdm=False),
):
    """Execute video upscaling pipeline and update progress bar."""
    if not input_file_path or not os.path.exists(input_file_path):
        raise gr.Error("Vui lòng chọn video đầu vào hợp lệ.")

    in_path = Path(input_file_path)
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{in_path.stem}_upscaled.mp4"

    # Scale mapping
    target_scale = 4 if "4x" in target_scale_choice else 2

    # Model resolution
    engine = None
    custom_fn = None
    model_name = "nomosuni"
    model_file = "models/2xNomosUni_compact_otf_medium.onnx"
    model_scale = 2

    if "ccsr" in model_choice.lower():
        # Check if CCSR engine is present on Colab
        try:
            from ecoup.engine.ccsr_engine import CcsrUpscaleEngine
            engine = CcsrUpscaleEngine()
            engine.load(model_path="models/ccsr_v2.pth", provider="cuda")
            model_scale = 2
            model_name = "ccsr_v2"
        except Exception as e:
            # Fallback to local high-sharpness model if CCSR weights not downloaded
            print(f"[Colab App] CCSR engine notice: {e}. Sử dụng NomosUni 2x sắc nét.")
            model_file = "models/2xNomosUni_compact_otf_medium.onnx"
            model_scale = 2
            model_name = "NomosUni_2x"
    elif "nomosuni" in model_choice.lower():
        model_file = "models/2xNomosUni_compact_otf_medium.onnx"
        model_scale = 2
        model_name = "NomosUni_2x"
    elif "span" in model_choice.lower():
        model_file = "models/2x-spanx2-ch48.onnx"
        model_scale = 2
        model_name = "SPAN_2x"
    elif "anime" in model_choice.lower():
        model_file = "models/realesr-animevideov3.onnx"
        model_scale = 4
        model_name = "realesr-animevideov3"
    elif "x2plus" in model_choice.lower():
        model_file = "models/2x-realesrgan-x2plus.onnx"
        model_scale = 2
        model_name = "RealESRGAN_x2plus"
    else:
        model_file = "models/realesr-general-x4v3.onnx"
        model_scale = 4
        model_name = "realesr-general-x4v3"

    if engine is None:
        mf = Path(model_file)
        if mf.exists():
            engine = OnnxUpscaleEngine()
            engine.load(mf, provider="auto")
            if hasattr(engine, "scale") and engine.scale > 0:
                model_scale = engine.scale
        else:
            # Fallback Lanczos4
            def fallback(tile):
                th, tw = tile.shape[:2]
                return cv2.resize(tile, (tw * model_scale, th * model_scale), interpolation=cv2.INTER_LANCZOS4)
            custom_fn = fallback

    cfg = AppConfig(
        model=ModelConfig(
            name=model_name,
            path=model_file,
            scale=model_scale,
            target_scale=target_scale,
        ),
        inference=InferenceConfig(
            provider="auto",
            tile_size=256,
            tile_overlap=16,
        ),
        thermal=ThermalConfig(enabled=False),  # Cloud GPU can run at full power
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
        last_status = (
            f"Đang xử lý: Khung hình {frame_idx}/{total_frames} ({int(pct * 100)}%) "
            f"• {fps:.1f} fps • Còn lại: {mins:02d}:{secs:02d} • Bỏ qua trùng: {dedup_rate:.1f}%"
        )
        progress(pct, desc=last_status)

    pipeline = UpscalePipeline(
        config=cfg,
        engine=engine,
        custom_upscale_fn=custom_fn,
        on_progress=on_progress,
    )

    progress(0.01, desc="Đang phân tích cấu trúc video...")
    pipeline.run(in_path, out_path, resume=False)
    progress(1.0, desc="Hoàn tất upscale! Video đã sẵn sàng tải về.")

    return str(out_path), f"Hoàn tất thành công! Đã lưu: {out_path.name}"


def create_ui():
    """Create minimalist, utilitarian Gradio interface."""
    custom_css = """
    #title-box { text-align: center; margin-bottom: 8px; }
    #title-box h1 { font-size: 1.5rem; margin-bottom: 4px; }
    #title-box p { color: #888; font-size: 0.9rem; }
    .gr-button-primary { background: #2563eb !important; }
    """

    with gr.Blocks(title="Eco Video Upscaler (Cloud Edition)", theme=gr.themes.Base(), css=custom_css) as demo:
        with gr.Column(elem_id="title-box"):
            gr.Markdown(
                """
                # ⚡ Eco Video Upscaler (Cloud CCSR Edition)
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
                    value="CCSR v2 (Content-Consistent - Khuyên dùng cho Cloud)",
                    label="Mô hình AI",
                )
                scale_dd = gr.Dropdown(
                    choices=["1080p (Scale 2x)", "1440p (Scale 2x)", "4K (Scale 4x)"],
                    value="1080p (Scale 2x)",
                    label="Độ phân giải đích",
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
            btn_run = gr.Button("🚀 BẮT ĐẦU UPSCALE", variant="primary", size="lg")
            status_output = gr.Textbox(
                label="Trạng thái xử lý",
                value="Sẵn sàng...",
                interactive=False,
                lines=1,
            )

        with gr.Group():
            gr.Markdown("### 4. Thành phẩm:")
            out_file = gr.File(
                label="Tệp video thành phẩm (Bấm vào đây để tải về thiết bị)",
                interactive=False,
            )

        btn_run.click(
            fn=run_upscale,
            inputs=[in_file, model_dd, scale_dd, denoise_sl, recover_sl, sharpen_sl, grain_cb],
            outputs=[out_file, status_output],
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
