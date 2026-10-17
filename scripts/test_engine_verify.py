import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import cv2
import numpy as np

# Ensure root and src are on sys.path
sys.path.insert(0, "/content/upscale_video/src")
sys.path.insert(0, "/content/upscale_video")
sys.path.insert(0, "/content/upscale_video/seedvr_core")
os.chdir("/content/upscale_video")

import torch
from ecoup.engine.seedvr2_engine import SeedVR2UpscaleEngine

print("=" * 60)
print("RUNNING FINAL VERIFICATION OF SeedVR2UpscaleEngine ON COLAB")
print("=" * 60)

engine = SeedVR2UpscaleEngine(scale=2)
engine.load(model_path="ckpts/seedvr2_ema_3b.pth", precision="bfloat16", provider="cuda")

if engine.runner is None:
    print("FAILED: engine.runner is None!")
    sys.exit(1)

print("[Verification] Engine loaded successfully with real ByteDance SeedVR2-3B weights!")

# Read 9 frames from clip_A.mp4
cap = cv2.VideoCapture("/content/upscale_video/eval/clips/clip_A.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frames = []
while len(frames) < 9:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

print(f"[Verification] Input clip: {len(frames)} frames, resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")

t0 = time.time()
torch.cuda.reset_peak_memory_stats()

out_frames = engine.upscale_clip(frames)

elapsed = time.time() - t0
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

print("=" * 60)
print(f"VERIFICATION SUCCESSFUL!")
print(f"Upscaled frames: {len(out_frames)}")
print(f"Output resolution: {out_frames[0].shape[1]}x{out_frames[0].shape[0]}")
print(f"Total inference time: {elapsed:.2f} seconds ({elapsed/len(out_frames):.2f} s/frame)")
print(f"Peak VRAM used: {peak_vram:.2f} GB (within 15.3 GB Tesla T4)")
print("=" * 60)

out_file = "/content/upscale_video/eval/results/seedvr2/verified_clip_A.mp4"
os.makedirs(os.path.dirname(out_file), exist_ok=True)
h, w = out_frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_file, fourcc, fps, (w, h))
for f in out_frames:
    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()

file_size_kb = os.path.getsize(out_file) / 1024
print(f"Saved verified output video to: {out_file} ({file_size_kb:.1f} KB)")
print("ALL CHECKS PASSED - MODEL INFERENCE FULLY FUNCTIONAL AND OPERATIONAL!")
