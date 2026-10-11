"""Command-line interface for Eco Video Upscaler."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from typing import Optional


import os


def _ensure_path_refreshed() -> None:
    """On Windows, re-read user/machine PATH from registry in case packages were just installed."""
    if sys.platform == "win32":
        try:
            import winreg
            paths = []
            for hkey, subkey in [
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                (winreg.HKEY_CURRENT_USER, r"Environment"),
            ]:
                with winreg.OpenKey(hkey, subkey) as key:
                    val, _ = winreg.QueryValueEx(key, "Path")
                    paths.append(val)
            combined = ";".join(paths)
            current = os.environ.get("PATH", "")
            for p in combined.split(";"):
                if p and p not in current:
                    current = f"{current};{p}"
            os.environ["PATH"] = current
        except Exception:
            pass


def check_ffmpeg() -> dict[str, any]:
    """Check FFmpeg and NVENC hardware encoder availability."""
    _ensure_path_refreshed()
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")

    info = {
        "ffmpeg": ffmpeg_bin is not None,
        "ffmpeg_path": ffmpeg_bin,
        "ffprobe": ffprobe_bin is not None,
        "ffprobe_path": ffprobe_bin,
        "version": None,
        "h264_nvenc": False,
        "hevc_nvenc": False,
    }

    if ffmpeg_bin:
        try:
            res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True, check=False)
            if res.stdout:
                first_line = res.stdout.splitlines()[0] if res.stdout.splitlines() else ""
                info["version"] = first_line
        except Exception:
            pass

        try:
            res = subprocess.run([ffmpeg_bin, "-encoders"], capture_output=True, text=True, check=False)
            if res.stdout:
                info["h264_nvenc"] = "h264_nvenc" in res.stdout
                info["hevc_nvenc"] = "hevc_nvenc" in res.stdout
        except Exception:
            pass

    return info


def check_gpu() -> dict[str, any]:
    """Check NVIDIA GPU status using pynvml or nvidia-smi."""
    info = {
        "available": False,
        "count": 0,
        "name": None,
        "driver_version": None,
        "vram_total_mb": None,
        "temp_c": None,
        "power_w": None,
        "method": None,
    }

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            try:
                pwr = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            except Exception:
                pwr = None
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode("utf-8")

            info.update({
                "available": True,
                "count": count,
                "name": name,
                "driver_version": driver,
                "vram_total_mb": round(mem.total / (1024 * 1024)),
                "temp_c": temp,
                "power_w": round(pwr, 1) if pwr is not None else None,
                "method": "pynvml",
            })
            pynvml.nvmlShutdown()
            return info
    except Exception:
        pass

    # Fallback to nvidia-smi command line check
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            res = subprocess.run(
                [smi, "--query-gpu=name,driver_version,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 4:
                    info.update({
                        "available": True,
                        "count": 1,
                        "name": parts[0],
                        "driver_version": parts[1],
                        "vram_total_mb": int(float(parts[2])),
                        "temp_c": int(float(parts[3])),
                        "power_w": float(parts[4]) if len(parts) > 4 and parts[4] != "[Not Supported]" else None,
                        "method": "nvidia-smi",
                    })
        except Exception:
            pass

    return info


def check_onnx() -> dict[str, any]:
    """Check ONNX Runtime and installed execution providers."""
    info = {
        "installed": False,
        "version": None,
        "providers": [],
        "dml_available": False,
        "cuda_available": False,
        "tensorrt_available": False,
    }

    try:
        import onnxruntime as ort

        info["installed"] = True
        info["version"] = ort.__version__
        providers = ort.get_available_providers()
        info["providers"] = providers
        info["dml_available"] = "DmlExecutionProvider" in providers
        info["cuda_available"] = "CUDAExecutionProvider" in providers
        info["tensorrt_available"] = "TensorrtExecutionProvider" in providers
    except ImportError:
        pass
    except Exception as e:
        info["error"] = str(e)

    return info


def run_doctor() -> int:
    """Print system diagnostics report."""
    print("==================================================")
    print(" Eco Video Upscaler (ecoup) - Environment Doctor ")
    print("==================================================")

    # 1. Platform
    print(f"\n[1] Python & OS:")
    print(f"  - Python Version : {sys.version.split()[0]}")
    print(f"  - Executable     : {sys.executable}")
    print(f"  - Platform       : {sys.platform}")

    # 2. FFmpeg
    ff = check_ffmpeg()
    print(f"\n[2] FFmpeg & Video Tools:")
    if ff["ffmpeg"]:
        print(f"  - FFmpeg         : [OK] Found at {ff['ffmpeg_path']}")
        if ff["version"]:
            print(f"  - Version        : {ff['version']}")
        print(f"  - FFprobe        : [{'OK' if ff['ffprobe'] else 'MISSING'}] {ff['ffprobe_path'] or ''}")
        print(f"  - NVENC H.264    : [{'OK' if ff['h264_nvenc'] else 'NOT SUPPORTED'}]")
        print(f"  - NVENC HEVC     : [{'OK' if ff['hevc_nvenc'] else 'NOT SUPPORTED'}]")
    else:
        print("  - FFmpeg         : [WARNING] ffmpeg not found in PATH!")
        print("                     Please install FFmpeg (e.g. winget install Gyan.FFmpeg)")

    # 3. GPU
    gpu = check_gpu()
    print(f"\n[3] NVIDIA GPU & Thermal Access:")
    if gpu["available"]:
        print(f"  - GPU Name       : [OK] {gpu['name']} (Count: {gpu['count']})")
        print(f"  - Driver Version : {gpu['driver_version']}")
        print(f"  - VRAM Total     : {gpu['vram_total_mb']} MB")
        print(f"  - Current Temp   : {gpu['temp_c']} °C")
        if gpu["power_w"] is not None:
            print(f"  - Current Power  : {gpu['power_w']} W")
        print(f"  - Monitor Method : {gpu['method']}")
    else:
        print("  - GPU            : [INFO] No NVIDIA GPU or telemetry accessible.")
        print("                     Software fallback will use CPU.")

    # 4. ONNX Runtime
    onnx = check_onnx()
    print(f"\n[4] AI Inference (ONNX Runtime):")
    if onnx["installed"]:
        print(f"  - ONNX Runtime   : [OK] Version {onnx['version']}")
        print(f"  - Providers      : {', '.join(onnx['providers'])}")
        print(f"  - DirectML EP    : [{'OK (DirectX 12 GPU)' if onnx['dml_available'] else 'NOT AVAILABLE'}]")
        print(f"  - CUDA EP        : [{'OK' if onnx['cuda_available'] else 'NOT AVAILABLE'}]")
        print(f"  - TensorRT EP    : [{'OK' if onnx['tensorrt_available'] else 'NOT AVAILABLE'}]")
    else:
        print("  - ONNX Runtime   : [WARNING] onnxruntime not installed in this environment.")
        print("                     Run: pip install onnxruntime-directml")

    print("\n==================================================")
    all_good = ff["ffmpeg"] and (gpu["available"] or True)
    if all_good:
        print(" Diagnostic status: READY FOR PIPELINE")
    else:
        print(" Diagnostic status: SOME COMPONENTS NEED SETUP")
    print("==================================================\n")
    return 0


def main(args: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Eco Video Upscaler CLI")
    subparsers = parser.add_subparsers(dest="command")

    # doctor command
    subparsers.add_parser("doctor", help="Inspect environment and hardware capability")

    # placeholder run command
    run_parser = subparsers.add_parser("run", help="Upscale a video file")
    run_parser.add_argument("-i", "--input", help="Path to input video file")
    run_parser.add_argument("-o", "--output", help="Path to output video file")
    run_parser.add_argument("-c", "--config", default="config.yaml", help="Path to config.yaml")
    run_parser.add_argument("--scale", type=int, help="Output target scale factor")
    run_parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    parsed = parser.parse_args(args)

    if parsed.command == "doctor":
        return run_doctor()
    elif parsed.command == "run":
        from pathlib import Path
        from ecoup.config import load_config
        from ecoup.engine.onnx_engine import OnnxUpscaleEngine
        from ecoup.pipeline import UpscalePipeline
        from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

        cfg = load_config(parsed.config)
        input_file = parsed.input or cfg.input
        output_file = parsed.output or cfg.output

        if not input_file:
            print("[ERROR] Input video path is required. Use -i <path> or set 'input' in config.yaml.")
            return 1
        if not output_file:
            # Default output: input_upscaled.mp4
            p = Path(input_file)
            output_file = str(p.parent / f"{p.stem}_upscaled{p.suffix}")

        if parsed.scale:
            cfg.model.target_scale = parsed.scale

        # Initialize ONNX engine if model exists
        engine = None
        model_path = Path(cfg.model.path)
        if model_path.exists():
            engine = OnnxUpscaleEngine()
            engine.load(model_path, precision=cfg.model.precision, provider=cfg.inference.provider)
        else:
            print(f"[ERROR] ONNX model not found at '{model_path}'.")
            print("Please place the model file in 'models/' or update config.yaml.")
            return 1

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[fps]:.1f} fps[/cyan]"),
            TextColumn("•"),
            TextColumn("[yellow]{task.fields[temp]}°C[/yellow]"),
            TextColumn("[blue]{task.fields[power]}W[/blue]"),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("Upscaling", total=100, fps=0.0, temp="--", power="--")

            def on_progress(frame_idx, total_frames, fps, eta_sec, metrics, dedup_rate):
                progress.update(
                    task_id,
                    completed=frame_idx,
                    total=total_frames,
                    fps=fps,
                    temp=str(metrics.temp_c) if metrics else "--",
                    power=str(round(metrics.power_w, 1)) if metrics and metrics.power_w else "--",
                )

            pipeline = UpscalePipeline(config=cfg, engine=engine, on_progress=on_progress)
            try:
                pipeline.run(input_path=input_file, output_path=output_file, resume=parsed.resume)
            finally:
                if engine:
                    engine.close()

        return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
