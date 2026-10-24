"""Phase 1 Verification Script: Checkpoint and Resume with SeedVR2 on Colab."""

import json
import os
import shutil
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, "/content/upscale_video/src")
sys.path.insert(0, "/content/upscale_video/seedvr_core")

from ecoup.config import AppConfig, ModelConfig, InferenceConfig, CheckpointConfig, EncodeConfig
from ecoup.pipeline import UpscalePipeline
from ecoup.engine.seedvr2_engine import SeedVR2UpscaleEngine
from ecoup.video_io import get_video_metadata

def main():
    input_video = Path("/content/upscale_video/plan_for_project/tests_video/720_A.mp4")
    if not input_video.exists():
        print(f"[ERROR] Input video not found: {input_video}")
        return 1

    meta_in = get_video_metadata(input_video)
    print(f"[Phase 1] Input video: {input_video.name}, {meta_in.width}x{meta_in.height}, {meta_in.total_frames} frames, FPS: {meta_in.fps}, audio: {meta_in.has_audio}")

    # Create a trimmed 20-frame test clip from 720_A.mp4 for fast verified execution
    trimmed_video = Path("/tmp/phase1_test_20f.mp4")
    if trimmed_video.exists():
        trimmed_video.unlink()

    import subprocess
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(input_video),
        "-vframes", "20",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(trimmed_video)
    ]
    subprocess.run(cmd, check=True)
    meta_trimmed = get_video_metadata(trimmed_video)
    print(f"[Phase 1] Trimmed test video: {meta_trimmed.total_frames} frames, {meta_trimmed.duration_seconds:.2f}s")

    chk_dir = Path("/tmp/phase1_checkpoint")
    if chk_dir.exists():
        shutil.rmtree(chk_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)

    output_video = Path("/tmp/phase1_output.mp4")
    if output_video.exists():
        output_video.unlink()

    # Step 1: Run with interval_frames=10, simulate interrupt after segment 0 (10 frames)
    print("\n--- Part 1: Running Segment 1 (10 frames) then interrupting ---")
    cfg1 = AppConfig(
        model=ModelConfig(scale=2, target_scale=2, precision="fp16"),
        inference=InferenceConfig(
            clip_length=5,
            clip_overlap=1,
            memory_mode="resident",
            seed=42,
        ),
        checkpoint=CheckpointConfig(
            enabled=True,
            interval_frames=10,
            dir=str(chk_dir),
        ),
        encode=EncodeConfig(codec="auto", copy_audio=True),
    )

    engine = SeedVR2UpscaleEngine(
        scale=2,
        seed=42,
        memory_mode="resident",
        precision="fp16",
    )
    pipeline1 = UpscalePipeline(config=cfg1, engine=engine)

    def on_progress_interrupt(frame, total, fps, eta, metrics, skip):
        print(f"  [Progress] Frame {frame}/{total} ({frame/total*100:.1f}%) - FPS: {fps:.2f}")
        if frame >= 10:
            print("  --> Triggering interruption after segment 0 (frame 10)...")
            pipeline1.cancel()

    pipeline1.on_progress = on_progress_interrupt
    pipeline1.run(input_path=trimmed_video, output_path=output_video, resume=False)

    from ecoup.checkpoint import CheckpointManager
    job_id = f"job_{trimmed_video.stem}"
    chk_mgr = CheckpointManager(checkpoint_dir=str(chk_dir), job_id=job_id)
    state_file = chk_mgr.state_file
    assert state_file.exists(), f"State file {state_file} must exist after segment 0"
    with open(state_file, "r") as f:
        state = json.load(f)
    print(f"  [Checkpoint State] Processed: {state.get('processed_frames')}, Segments: {state.get('segments')}, Completed: {state.get('completed')}")
    assert state.get("processed_frames") >= 10, "Should have processed at least 10 frames"
    assert len(state.get("segments", [])) >= 1, "Should have at least 1 segment file"

    # Step 2: Resume pipeline to finish remaining frames
    print("\n--- Part 2: Resuming from Checkpoint to finish video ---")
    pipeline2 = UpscalePipeline(config=cfg1, engine=engine)
    def on_progress_resume(frame, total, fps, eta, metrics, skip):
        print(f"  [Resume Progress] Frame {frame}/{total} ({frame/total*100:.1f}%) - FPS: {fps:.2f}")
    pipeline2.on_progress = on_progress_resume

    # Check state before running pipeline2
    st = chk_mgr.load_state()
    print("State before resume:", st)
    for s in st.get("segments", []):
        meta = get_video_metadata(s)
        print(f"  Seg {s}: meta frames={meta.total_frames}")

    pipeline2.run(input_path=trimmed_video, output_path=output_video, resume=True)

    # Verification
    print("\n--- Part 3: Verification of Final Video ---")
    assert output_video.exists(), f"Output video {output_video} does not exist!"
    meta_res = get_video_metadata(output_video)
    print(f"  Output video: {meta_res.width}x{meta_res.height}, {meta_res.total_frames} frames, FPS: {meta_res.fps}")
    print(f"  Audio present: {meta_res.has_audio}, Audio codec: {meta_res.audio_codec}")

    # Check frame count
    assert meta_res.total_frames == meta_trimmed.total_frames, f"Frame count mismatch: {meta_res.total_frames} vs {meta_trimmed.total_frames}"
    assert meta_res.has_audio is True, "Audio was not preserved!"

    # Check sidecar metadata file
    sidecar_json = output_video.with_suffix(".json")
    assert sidecar_json.exists(), f"Sidecar JSON metadata {sidecar_json} was not created!"
    with open(sidecar_json, "r", encoding="utf-8") as f:
        meta_json_data = json.load(f)
    print(f"  Sidecar JSON: seed={meta_json_data.get('seed')}, memory_mode={meta_json_data.get('memory_mode')}")
    assert meta_json_data.get("seed") == 42
    assert meta_json_data.get("memory_mode") == "resident"

    print("\n[SUCCESS] Phase 1 Checkpoint & Resume verification passed 100%!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
