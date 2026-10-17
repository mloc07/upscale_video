"""Evaluation runner for SeedVR2-3B on Google Colab (Tesla T4).
Processes clip_A, clip_B, clip_C sequentially with alternating GPU offload,
VAE memory flushing, dynamic RoPE, and metric recording.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import psutil
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

seedvr_dir = "/content/upscale_video/seedvr_core"
if seedvr_dir not in sys.path:
    sys.path.insert(0, seedvr_dir)
os.chdir(seedvr_dir)

from common.config import load_config
from common.diffusion.utils import classifier_free_guidance_dispatcher
from common.distributed import get_device, init_torch
from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange
from einops import rearrange
import mediapy
import models.dit_v2.na as na
from projects.video_diffusion_sr.infer import VideoDiffusionInfer
from torchvision.transforms import Compose, Lambda, Normalize


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def clear_vae_memory(vae):
    """Clear all lingering causal activation memory buffers."""
    count = 0
    for m in vae.modules():
        if hasattr(m, "memory") and m.memory is not None:
            m.memory = None
            count += 1
    return count


def load_clip_frames(clip_path: str, max_frames: int = 9):
    """Load video frames using OpenCV."""
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def main():
    print("=" * 60, flush=True)
    print("Starting SeedVR2-3B Evaluation Benchmark Pipeline", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Initial VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)
    print("=" * 60, flush=True)

    init_torch(cudnn_benchmark=False)
    config = load_config("./configs_3b/main.yaml")
    runner = VideoDiffusionInfer(config)

    print("--> Loading DiT 3B checkpoint to CPU...", flush=True)
    runner.configure_dit_model(device="cpu", checkpoint="./ckpts/seedvr2_ema_3b.pth")
    clear_gpu()

    print("--> Loading VAE checkpoint to CPU...", flush=True)
    runner.configure_vae_model()
    runner.vae.to("cpu")
    clear_gpu()

    # Diffusion sampling config
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()

    # Pre-computed prompt embeddings
    pos_emb = torch.load("pos_emb.pt", map_location="cpu")
    neg_emb = torch.load("neg_emb.pt", map_location="cpu")
    text_embeds = {
        "texts_pos": [pos_emb.to(get_device())],
        "texts_neg": [neg_emb.to(get_device())],
    }
    del pos_emb, neg_emb
    clear_gpu()

    clips = ["clip_A", "clip_B", "clip_C"]
    clips_dir = Path("/content/upscale_video/eval/clips")
    out_dir = Path("/content/upscale_video/eval/results/seedvr2")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary = {}

    for clip_name in clips:
        clip_path = str(clips_dir / f"{clip_name}.mp4")
        if not os.path.exists(clip_path):
            print(f"[Warning] Clip not found: {clip_path}, skipping.", flush=True)
            continue

        print("\n" + "=" * 55, flush=True)
        print(f"Processing {clip_name}...", flush=True)
        print("=" * 55, flush=True)

        frames, fps = load_clip_frames(clip_path, max_frames=9)
        num_frames = len(frames)
        print(f"Loaded {num_frames} frames at {fps:.1f} fps.", flush=True)

        video_np = np.stack(frames, axis=0)
        orig_h, orig_w = video_np.shape[1], video_np.shape[2]
        res_h, res_w = 1280, 720

        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
        transform = Compose([
            NaResize(resolution=(res_h * res_w) ** 0.5, mode="area", downsample_only=False),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            DivisibleCrop((16, 16)),
            Normalize(0.5, 0.5),
            Rearrange("t c h w -> c t h w"),
        ])
        cond_video = transform(video_tensor).unsqueeze(0).to(torch.bfloat16)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        # Step 1: VAE Encode on GPU
        print("--> 1. VAE Encoding on GPU...", flush=True)
        runner.vae.to(device=get_device(), dtype=torch.bfloat16)
        with torch.no_grad():
            cond_latents = runner.vae_encode([cond_video.squeeze(0)])

        clear_vae_memory(runner.vae)
        runner.vae.to("cpu")
        clear_gpu()

        # Step 2: DiT Sampling on GPU
        print("--> 2. DiT Sampling on GPU...", flush=True)
        runner.dit.to(device=get_device(), dtype=torch.bfloat16)

        noises = [torch.randn_like(latent).to(get_device()) for latent in cond_latents]
        cond_latents_cuda = [c.to(get_device()) for c in cond_latents]
        conditions = [
            runner.get_condition(noise, task="sr", latent_blur=latent_blur)
            for noise, latent_blur in zip(noises, cond_latents_cuda)
        ]

        latents, latents_shapes = na.flatten(noises)
        latents_cond, _ = na.flatten(conditions)
        text_pos_embeds, text_pos_shapes = na.flatten(text_embeds["texts_pos"])

        was_training = runner.dit.training
        runner.dit.eval()

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            latents = runner.sampler.sample(
                x=latents,
                f=lambda args: classifier_free_guidance_dispatcher(
                    pos=lambda: runner.dit(
                        vid=torch.cat([args.x_t, latents_cond], dim=-1),
                        txt=text_pos_embeds,
                        vid_shape=latents_shapes,
                        txt_shape=text_pos_shapes,
                        timestep=args.t.repeat(len(noises)),
                    ).vid_sample,
                    neg=lambda: None,
                    scale=1.0,
                    rescale=0.0,
                ),
            )

        runner.dit.train(was_training)
        latents = na.unflatten(latents, latents_shapes)

        # Step 3: Offload DiT to CPU to give 100% free VRAM to VAE Decode
        print("--> 3. Offloading DiT to CPU...", flush=True)
        runner.dit.to("cpu")
        clear_gpu()

        # Step 4: VAE Decode on GPU
        print("--> 4. VAE Decoding on GPU...", flush=True)
        runner.vae.to(device=get_device(), dtype=torch.bfloat16)
        with torch.no_grad():
            video_tensors = runner.vae_decode(latents)

        clear_vae_memory(runner.vae)
        runner.vae.to("cpu")
        clear_gpu()

        total_time = time.time() - t0
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        fps_speed = num_frames / total_time

        print(f"--> Done {clip_name} in {total_time:.2f}s ({fps_speed:.2f} fps) | Peak VRAM: {peak_vram_gb:.2f} GB", flush=True)

        # Output video formatting
        out = video_tensors[0].to("cpu")
        if out.ndim == 4:
            out = out.permute(1, 2, 3, 0)
        out_np = out.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round().to(torch.uint8).numpy()

        out_file = str(out_dir / f"{clip_name}_seedvr2.mp4")
        mediapy.write_video(out_file, out_np, fps=fps)
        out_size_kb = os.path.getsize(out_file) / 1024
        print(f"--> Saved output to: {out_file} ({out_size_kb:.1f} KB)", flush=True)

        clip_metric = {
            "clip": clip_name,
            "frames": num_frames,
            "resolution_in": f"{orig_w}x{orig_h}",
            "resolution_out": f"{out_np.shape[2]}x{out_np.shape[1]}",
            "time_sec": round(total_time, 2),
            "fps": round(fps_speed, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "output_file": out_file,
        }
        metrics_summary[clip_name] = clip_metric

        metric_path = out_dir / f"metrics_{clip_name}.json"
        with open(metric_path, "w", encoding="utf-8") as f:
            json.dump(clip_metric, f, indent=2)

    all_metrics_path = out_dir / "benchmark_metrics.json"
    with open(all_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("ALL EVALUATION CLIPS PROCESSED SUCCESSFULLY!", flush=True)
    print(json.dumps(metrics_summary, indent=2), flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
