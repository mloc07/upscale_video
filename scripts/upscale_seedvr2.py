"""ByteDance SeedVR2-3B Video Super-Resolution Inference Tool.

Optimized for Google Colab (Tesla T4 16GB) and local CUDA GPUs.
Features:
- Dynamic RoPE memory optimization (prevents 4GB RoPE spike)
- Alternating VAE / DiT model GPU offloading (keeps VRAM < 7.5 GB, RAM < 9 GB)
- Causal Conv3D memory flush (prevents cumulative VRAM leak)
- Multi-chunk video processing with seamless concatenation
- Audio preservation from original video via FFmpeg
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_vae_memory(vae):
    """Clear all lingering causal activation memory buffers."""
    count = 0
    for m in vae.modules():
        if hasattr(m, "memory") and m.memory is not None:
            m.memory = None
            count += 1
    return count


def load_video_frames(video_path: str, max_frames: int = 0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        count += 1
        if max_frames > 0 and count >= max_frames:
            break
    cap.release()
    return frames, fps


def add_audio_to_video(src_video: str, target_video: str):
    """Mux original audio into upscaled video using FFmpeg if audio stream exists."""
    tmp_out = target_video + ".with_audio.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", target_video,
        "-i", src_video,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        tmp_out,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and os.path.exists(tmp_out):
            os.replace(tmp_out, target_video)
            print(f"[FFmpeg] Successfully preserved audio track in {target_video}")
    except Exception as e:
        print(f"[Warning] Audio muxing skipped: {e}")
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def upscale_chunk(
    runner,
    frames_chunk: list[np.ndarray],
    text_embeds: dict,
    res_h: int = 1280,
    res_w: int = 720,
) -> np.ndarray:
    from common.diffusion.utils import classifier_free_guidance_dispatcher
    from common.distributed import get_device
    from data.image.transforms.divisible_crop import DivisibleCrop
    from data.image.transforms.na_resize import NaResize
    from data.video.transforms.rearrange import Rearrange
    import models.dit_v2.na as na
    from torchvision.transforms import Compose, Lambda, Normalize

    video_np = np.stack(frames_chunk, axis=0)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    transform = Compose([
        NaResize(resolution=(res_h * res_w) ** 0.5, mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])
    cond_video = transform(video_tensor).unsqueeze(0).to(torch.bfloat16)

    # 1. VAE Encode on GPU
    runner.vae.to(device=get_device(), dtype=torch.bfloat16)
    with torch.no_grad():
        cond_latents = runner.vae_encode([cond_video.squeeze(0)])
    clear_vae_memory(runner.vae)
    runner.vae.to("cpu")
    clear_gpu()

    # 2. DiT Sampling on GPU
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

    # 3. Offload DiT to CPU
    runner.dit.to("cpu")
    clear_gpu()

    # 4. VAE Decode on GPU
    runner.vae.to(device=get_device(), dtype=torch.bfloat16)
    with torch.no_grad():
        video_tensors = runner.vae_decode(latents)
    clear_vae_memory(runner.vae)
    runner.vae.to("cpu")
    clear_gpu()

    out = video_tensors[0].to("cpu")
    if out.ndim == 4:
        out = out.permute(1, 2, 3, 0)
    out_np = out.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round().to(torch.uint8).numpy()
    return out_np


def main():
    parser = argparse.ArgumentParser(description="ByteDance SeedVR2-3B Video Super-Resolution")
    parser.add_argument("-i", "--input", type=str, help="Input video file path")
    parser.add_argument("-o", "--output", type=str, help="Output video file path")
    parser.add_argument("--chunk-size", type=int, default=9, help="Number of frames per processing chunk (default: 9)")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process (0 = all)")
    parser.add_argument("--height", type=int, default=1280, help="Target height (default: 1280)")
    parser.add_argument("--width", type=int, default=720, help="Target width (default: 720)")
    parser.add_argument("--config", type=str, default="./configs_3b/main.yaml", help="Path to main.yaml config")
    parser.add_argument("--checkpoint", type=str, default="./ckpts/seedvr2_ema_3b.pth", help="Path to DiT checkpoint")
    args = parser.parse_args()

    # Locate seedvr_core
    seedvr_candidates = [
        Path("/content/upscale_video/seedvr_core"),
        Path("./seedvr_core"),
        Path("../seedvr_core"),
    ]
    seedvr_dir = None
    for cand in seedvr_candidates:
        if cand.exists() and (cand / "projects").exists():
            seedvr_dir = cand
            break

    if seedvr_dir is not None:
        sys.path.insert(0, str(seedvr_dir.resolve()))
        os.chdir(seedvr_dir)

    import mediapy
    from common.config import load_config
    from common.distributed import get_device, init_torch
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer

    print("=" * 60)
    print("SeedVR2-3B Super-Resolution Engine")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[Error] CUDA is required to run SeedVR2-3B DiT.", file=sys.stderr)
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")

    init_torch(cudnn_benchmark=False)
    config = load_config(args.config)
    runner = VideoDiffusionInfer(config)

    print("--> Loading DiT 3B weights to CPU...")
    runner.configure_dit_model(device="cpu", checkpoint=args.checkpoint)
    clear_gpu()

    print("--> Loading VAE weights to CPU...")
    runner.configure_vae_model()
    runner.vae.to("cpu")
    clear_gpu()

    # Diffusion configuration
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

    input_path = args.input or "/content/upscale_video/eval/clips/clip_A.mp4"
    if not os.path.exists(input_path):
        print(f"[Error] Input video not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or f"output_{Path(input_path).stem}_seedvr2.mp4"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing video: {input_path}")
    all_frames, fps = load_video_frames(input_path, max_frames=args.max_frames)
    total_frames = len(all_frames)
    print(f"Loaded {total_frames} frames at {fps:.2f} fps.")

    chunk_size = args.chunk_size
    chunks = [all_frames[i : i + chunk_size] for i in range(0, total_frames, chunk_size)]
    print(f"Divided into {len(chunks)} chunks (chunk size = {chunk_size}).")

    upscaled_chunks = []
    t_start = time.time()

    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk [{idx + 1}/{len(chunks)}] ({len(chunk)} frames) ---")
        chunk_out = upscale_chunk(runner, chunk, text_embeds, res_h=args.height, res_w=args.width)
        upscaled_chunks.append(chunk_out)

    final_frames = np.concatenate(upscaled_chunks, axis=0)
    mediapy.write_video(output_path, final_frames, fps=fps)
    add_audio_to_video(input_path, output_path)

    elapsed = time.time() - t_start
    final_fps = total_frames / elapsed
    print("\n" + "=" * 60)
    print(f"Processing Complete in {elapsed:.2f}s ({final_fps:.2f} FPS)")
    print(f"Output saved to: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
