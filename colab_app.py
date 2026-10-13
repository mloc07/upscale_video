"""Gradio Web Application for Eco Video Upscaler (Optimized for Google Colab and remote devices).

Giao diện phục chế và nâng cấp độ phân giải video AI:
- Thiết kế tối giản, tập trung vào công việc
- Hỗ trợ mô hình khuếch tán CCSR v2 và các mô hình ONNX gia tốc GPU
- Hỗ trợ hủy tiến trình tức thì
- Giám sát trạng thái và tài nguyên thời gian thực
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

# Avoid UnicodeEncodeError on Windows consoles (cp1252)
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

# Verified ONNX models mapping: model_key -> (filename, download_url, default_scale)
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
        "CCSR v2 (Diffusion Model)",
        "RealESRGAN x2plus (2x)",
        "NomosUni 2x",
        "RealESRGAN 4x (General)",
        "RealESRGAN 4x (Anime)",
        "SPAN 2x",
    ]


def ensure_model_file(model_key: str) -> tuple[Path, int]:
    """Ensure ONNX model file exists on disk, downloading if necessary."""
    if model_key not in MODEL_REGISTRY:
        model_key = "nomosuni"
    filename, url, scale = MODEL_REGISTRY[model_key]
    dest = Path("models") / filename
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Model Loader] Đang tải mô hình {filename} từ {url}...")
        urllib.request.urlretrieve(url, str(dest))
    return dest, scale


def cancel_upscale():
    """Cancel currently running upscale job."""
    global active_pipeline
    if active_pipeline is not None:
        active_pipeline.cancel()
        return "Đã gửi lệnh hủy. Đang dừng an toàn và giải phóng GPU..."
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
    """Execute video upscaling pipeline."""
    global active_pipeline

    if not input_file_path or not os.path.exists(input_file_path):
        raise gr.Error("Vui lòng chọn video đầu vào hợp lệ.")

    in_path = Path(input_file_path)
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{in_path.stem}_upscaled.mp4"

    # Scale mapping
    target_scale = 4 if "4" in str(target_scale_choice) else 2

    # Tile mapping
    if "toàn khung" in tile_choice.lower() or "full" in tile_choice.lower():
        tile_sz = 0
        tile_ov = 0
    elif "256" in tile_choice:
        tile_sz = 256
        tile_ov = 16
    else:  # 512
        tile_sz = 512
        tile_ov = 16

    progress(0.01, desc="Đang chuẩn bị mô hình AI...")

    # Model resolution
    engine = None
    custom_fn = None
    model_name = "nomosuni"
    model_scale = 2
    model_file_str = "models/model.onnx"

    if "ccsr" in model_choice.lower():
        from ecoup.engine.ccsr_engine import CcsrUpscaleEngine
        engine = CcsrUpscaleEngine()
        model_name = "CCSR_v2"
        model_scale = 2
        engine.load(model_path="preset/models", precision="fp16", provider="cuda")
        model_file_str = "preset/models/controlnet"
    elif "x2plus" in model_choice.lower():
        mf, model_scale = ensure_model_file("x2plus")
        model_name = "RealESRGAN_x2plus"
        model_file_str = str(mf)
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    elif "anime" in model_choice.lower():
        mf, model_scale = ensure_model_file("anime")
        model_name = "realesr-animevideov3"
        model_file_str = str(mf)
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    elif "span" in model_choice.lower():
        mf, model_scale = ensure_model_file("span")
        model_name = "SPAN_2x"
        model_file_str = str(mf)
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    elif "general" in model_choice.lower():
        mf, model_scale = ensure_model_file("general")
        model_name = "realesr-general-x4v3"
        model_file_str = str(mf)
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")
    else:
        mf, model_scale = ensure_model_file("nomosuni")
        model_name = "NomosUni_2x"
        model_file_str = str(mf)
        engine = OnnxUpscaleEngine()
        engine.load(mf, provider="auto")

    if engine is not None and hasattr(engine, "scale") and engine.scale > 0:
        model_scale = engine.scale

    cfg = AppConfig(
        model=ModelConfig(
            name=model_name,
            path=model_file_str,
            scale=model_scale,
            target_scale=target_scale,
        ),
        inference=InferenceConfig(
            provider="auto",
            tile_size=tile_sz,
            tile_overlap=tile_ov,
        ),
        thermal=ThermalConfig(enabled=False),
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

        # Telemetry
        used_gb, total_gb, gpu_name = get_gpu_vram_info()
        vram_info = ""
        if total_gb > 0:
            vram_pct = (used_gb / total_gb) * 100
            vram_info = f" • VRAM: {used_gb:.1f}/{total_gb:.1f}GB ({vram_pct:.0f}%)"

        prov = getattr(engine, "active_provider", "GPU")

        last_status = (
            f"Khung hình {frame_idx}/{total_frames} ({int(pct * 100)}%) "
            f"• {fps:.1f} fps • Còn lại: {mins:02d}:{secs:02d}{vram_info} • {prov}"
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
        progress(0.02, desc="Đang phân tích video...")
        pipeline.run(in_path, out_path, resume=False)
    except Exception as e:
        if pipeline._cancelled:
            return None, "Quá trình xử lý đã được hủy thành công."
        raise e
    finally:
        active_pipeline = None

    if pipeline._cancelled:
        return None, "Quá trình xử lý đã được hủy thành công."

    progress(1.0, desc="Hoàn tất xử lý.")
    return str(out_path), f"Hoàn tất. Video đã lưu: {out_path.name}"


def create_ui():
    """Create clean, modern Gradio interface."""
    custom_css = """
    .gradio-container { max-width: 900px !important; margin: 0 auto !important; }
    #header { text-align: center; margin-bottom: 16px; }
    #header h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }
    #header p { color: #64748b; font-size: 0.95rem; margin: 0; }
    .gr-button-primary { background: #2563eb !important; }
    .gr-button-stop { background: #dc2626 !important; color: white !important; }
    """

    with gr.Blocks(title="Eco Video Upscaler") as demo:
        with gr.Column(elem_id="header"):
            gr.Markdown(
                """
                # ⚡ Eco Video Upscaler
                Hệ thống phục chế và nâng cấp độ phân giải video AI
                """
            )

        with gr.Group():
            gr.Markdown("#### 1. Tệp nguồn")
            in_file = gr.File(
                label="Tải lên Video",
                file_types=[".mp4", ".mov", ".mkv", ".avi", ".webm"],
                type="filepath",
            )

        with gr.Group():
            gr.Markdown("#### 2. Cấu hình xử lý")
            with gr.Row():
                model_dd = gr.Dropdown(
                    choices=get_available_models(),
                    value="CCSR v2 (Diffusion Model)",
                    label="Mô hình AI",
                )
                scale_dd = gr.Dropdown(
                    choices=["2x", "4x"],
                    value="2x",
                    label="Tỉ lệ phóng đại",
                )
                tile_dd = gr.Dropdown(
                    choices=[
                        "Toàn khung (Full-Frame)",
                        "Tile 512x512",
                        "Tile 256x256",
                    ],
                    value="Toàn khung (Full-Frame)",
                    label="Phân mảnh (Tile)",
                )

            with gr.Row():
                denoise_sl = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=30,
                    step=5,
                    label="Khử nhiễu (Denoise %)",
                )
                recover_sl = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Khôi phục gốc (Recover %)",
                )

            with gr.Row():
                sharpen_sl = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=20,
                    step=5,
                    label="Làm nét (Sharpen %)",
                )
                grain_cb = gr.Checkbox(
                    value=True,
                    label="Bù hạt phim (Film Grain)",
                )

        with gr.Group():
            gr.Markdown("#### 3. Điều khiển")
            with gr.Row():
                btn_run = gr.Button("Bắt đầu xử lý", variant="primary", size="lg", scale=3)
                btn_cancel = gr.Button("Hủy bỏ", variant="stop", size="lg", scale=1)

            status_output = gr.Textbox(
                label="Trạng thái",
                value="Sẵn sàng",
                interactive=False,
                lines=1,
            )

        with gr.Group():
            gr.Markdown("#### 4. Kết quả")
            out_file = gr.File(
                label="Video thành phẩm",
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
    print("🚀 Khởi động Eco Video Upscaler WebUI...")
    print("=======================================================\n")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
