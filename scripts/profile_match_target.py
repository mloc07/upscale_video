"""Profile and compare native vs match_target internal resolution on Colab L4."""

import json
import os
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, "/content/upscale_video/src")
sys.path.insert(0, "/content/upscale_video/seedvr_core")

from ecoup.engine.seedvr2_engine import SeedVR2UpscaleEngine
from ecoup.video_io import VideoReader, get_video_metadata

def run_test(res_mode: str, frames: list):
    print(f"\n==========================================")
    print(f"Testing SeedVR2 with model_resolution = '{res_mode}'")
    print(f"==========================================")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    engine = SeedVR2UpscaleEngine(
        scale=2,
        seed=42,
        memory_mode="resident",
        precision="fp16",
        sub_len=5,
        model_resolution=res_mode,
    )
    engine.load(precision="fp16")
    t0 = time.time()
    try:
        out_frames = engine.upscale_clip(frames)
        elapsed = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        h, w = out_frames[0].shape[:2]
        print(f"  Result: Success!")
        print(f"  Output Frame Shape: {w}x{h}")
        print(f"  Elapsed Time (5 frames): {elapsed:.2f}s ({5/elapsed:.2f} FPS)")
        print(f"  Peak VRAM: {peak_vram:.2f} GB")
        if hasattr(engine, "_step_timer") and engine._step_timer:
            print("  Stage timings:")
            for stage, t in engine._step_timer.summary().items():
                print(f"    - {stage}: {t:.2f}s")
        return {
            "status": "success",
            "resolution": f"{w}x{h}",
            "elapsed": elapsed,
            "fps": 5 / elapsed,
            "peak_vram_gb": peak_vram,
            "timings": engine._step_timer.summary() if hasattr(engine, "_step_timer") else {},
        }
    except Exception as exc:
        print(f"  Result: Failed with {type(exc).__name__}: {exc}")
        return {
            "status": "failed",
            "error": str(exc),
        }

def main():
    input_video = "/content/upscale_video/plan_for_project/tests_video/720_A.mp4"
    vr = VideoReader(input_video)
    frames = []
    for idx, f in enumerate(vr):
        frames.append(f)
        if len(frames) >= 5:
            break
    vr.close()
    print(f"Loaded {len(frames)} frames of shape {frames[0].shape}")

    res_native = run_test("native", frames)
    res_match = run_test("match_target", frames)

    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"Native (720p internal + 2x resize):")
    print(f"  VRAM: {res_native.get('peak_vram_gb', 'N/A')} GB | Time: {res_native.get('elapsed', 'N/A')}s ({res_native.get('fps', 'N/A')} FPS)")
    print(f"Match Target (1440x2560 direct internal):")
    if res_match.get("status") == "success":
        print(f"  VRAM: {res_match.get('peak_vram_gb', 'N/A')} GB | Time: {res_match.get('elapsed', 'N/A')}s ({res_match.get('fps', 'N/A')} FPS)")
    else:
        print(f"  Status: {res_match.get('status')} - Error: {res_match.get('error')}")

if __name__ == "__main__":
    main()
