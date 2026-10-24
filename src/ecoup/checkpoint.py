"""Checkpoint and segment management for interruptible, resumable upscaling jobs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ecoup.cli import _ensure_path_refreshed


class CheckpointManager:
    """Manages segment video files and state for resuming interrupted jobs."""

    def __init__(self, checkpoint_dir: str | Path, job_id: str = "job_default"):
        self.checkpoint_dir = Path(checkpoint_dir).resolve() / job_id
        self.state_file = self.checkpoint_dir / "state.json"
        self.segments_dir = self.checkpoint_dir / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> dict:
        """Load state from JSON or return default empty state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "processed_frames": 0,
            "segments": [],
            "completed": False,
        }

    def save_state(self, processed_frames: int, segments: list[str], completed: bool = False) -> None:
        """Persist current state to JSON."""
        state = {
            "processed_frames": processed_frames,
            "segments": segments,
            "completed": completed,
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def get_segment_path(self, segment_index: int) -> Path:
        """Get output path for a specific segment index."""
        return self.segments_dir / f"segment_{segment_index:05d}.mp4"

    def concat_segments(
        self,
        segment_paths: list[str | Path],
        final_output_path: str | Path,
        audio_source: Optional[str | Path] = None,
    ) -> None:
        """Stitch video segments together using FFmpeg concat demuxer and remux audio."""
        _ensure_path_refreshed()
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg not found for segment concatenation.")

        if not segment_paths:
            raise ValueError("No segments to concatenate.")

        final_out = Path(final_output_path).resolve()
        os.makedirs(final_out.parent, exist_ok=True)

        # Write concat list file
        concat_txt = self.checkpoint_dir / "concat_list.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in segment_paths:
                clean_p = Path(p).resolve().as_posix()
                f.write(f"file '{clean_p}'\n")

        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_txt),
        ]

        if audio_source and os.path.exists(str(audio_source)):
            cmd.extend([
                "-i", str(Path(audio_source).resolve()),
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "copy",
                "-c:a", "copy",
            ])
        else:
            cmd.extend([
                "-map", "0:v:0",
                "-c:v", "copy",
            ])

        cmd.append(str(final_out))

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {res.stderr}")

    def cleanup(self) -> None:
        """Remove temporary segments and checkpoint directory after successful completion."""
        if self.checkpoint_dir.exists():
            try:
                shutil.rmtree(self.checkpoint_dir)
            except Exception:
                pass
