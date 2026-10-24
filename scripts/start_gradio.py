"""Launcher for Eco Video Upscaler Gradio WebUI on Colab with Cloudflare Tunnel & Gradio Share."""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

def main():
    print("[1/4] Dọn dẹp tiến trình cũ nếu có...")
    subprocess.run("pkill -f 'colab_app.py' || true", shell=True)
    subprocess.run("pkill -f 'cloudflared.*7860' || true", shell=True)
    time.sleep(1)

    log_gradio = Path("/tmp/gradio.log")
    log_cf = Path("/tmp/gradio_cf.log")
    if log_gradio.exists():
        log_gradio.unlink()
    if log_cf.exists():
        log_cf.unlink()

    print("[2/4] Đang khởi chạy Eco Video Upscaler Gradio WebUI trên cổng 7860...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/content/upscale_video/src:/content/upscale_video/seedvr_core:" + env.get("PYTHONPATH", "")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd_gradio = [
        sys.executable,
        "/content/upscale_video/colab_app.py",
        "--port", "7860",
        "--host", "0.0.0.0",
        "--share",
    ]
    with open(log_gradio, "w") as out:
        p_gradio = subprocess.Popen(
            cmd_gradio,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/content/upscale_video",
        )

    print("[3/4] Đang khởi tạo Cloudflare Tunnel cho WebUI...")
    cmd_cf = [
        "cloudflared",
        "tunnel",
        "--url", "http://127.0.0.1:7860",
        "--logfile", str(log_cf),
    ]
    p_cf = subprocess.Popen(cmd_cf, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[4/4] Đang chờ liên kết công khai (Public URL)...")
    cf_url = None
    gradio_url = None

    for i in range(25):
        time.sleep(1)
        # Check CF URL
        if not cf_url and log_cf.exists():
            content = log_cf.read_text(errors="ignore")
            m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", content)
            if m:
                cf_url = m.group(0)

        # Check Gradio Share URL
        if not gradio_url and log_gradio.exists():
            content = log_gradio.read_text(errors="ignore")
            m = re.search(r"https://[-a-z0-9]+\.gradio\.live", content)
            if m:
                gradio_url = m.group(0)

        if cf_url or gradio_url:
            break

    print("\n" + "=" * 60)
    print("🚀 GRADIO WEBUI ĐÃ KHỞI CHẠY THÀNH CÔNG!")
    print("=" * 60)
    if cf_url:
        print(f"🔗 Cloudflare Tunnel Link (Khuyên dùng - Tốc độ cao):")
        print(f"   {cf_url}")
    if gradio_url:
        print(f"🔗 Gradio Live Share Link:")
        print(f"   {gradio_url}")
    print(f"🏠 Localhost: http://127.0.0.1:7860")
    print("=" * 60)

    # Print log snippet
    if log_gradio.exists():
        print("\n[Log khởi động Gradio]:")
        lines = log_gradio.read_text(errors="ignore").strip().splitlines()
        for l in lines[-10:]:
            print("  ", l)

if __name__ == "__main__":
    main()
