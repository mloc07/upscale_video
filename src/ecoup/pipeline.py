"""Main orchestration pipeline for Eco Video Upscaler."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np

from ecoup.checkpoint import CheckpointManager
from ecoup.config import AppConfig
from ecoup.dedup import FrameDeduplicator
from ecoup.engine.base import UpscaleEngine
from ecoup.thermal import ThermalGovernor
from ecoup.tiling import upscale_with_tiling
from ecoup.utils import setup_logger, format_duration
from ecoup.video_io import VideoReader, VideoWriter, get_video_metadata


class UpscalePipeline:
    """Coordinates VideoReader, Dedup, ThermalGovernor, AI Inference, and VideoWriter."""

    def __init__(
        self,
        config: AppConfig,
        engine: Optional[UpscaleEngine] = None,
        custom_upscale_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        on_progress: Optional[Callable[[int, int, float, float, any, float], None]] = None,
    ):
        self.config = config
        self.engine = engine
        self.custom_upscale_fn = custom_upscale_fn
        self.on_progress = on_progress
        self.logger = setup_logger("ecoup")
        self._cancelled = False
        self._paused = False

    def cancel(self) -> None:
        """Signal pipeline to stop after the current frame."""
        self._cancelled = True

    def set_paused(self, paused: bool) -> None:
        """Pause or resume pipeline processing."""
        self._paused = paused

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        resume: bool = False,
    ) -> None:
        input_file = Path(input_path).resolve()
        output_file = Path(output_path).resolve()

        if not input_file.exists():
            raise FileNotFoundError(f"Input video not found: {input_file}")

        meta = get_video_metadata(input_file)
        self.logger.info(
            f"Input video: {meta.width}x{meta.height} @ {meta.fps:.2f} fps ({meta.total_frames} frames), Audio: {meta.has_audio}"
        )

        if self.engine is not None and hasattr(self.engine, "scale") and self.engine.scale > 0:
            scale = self.engine.scale
        else:
            scale = self.config.model.scale
        target_scale = float(self.config.model.target_scale)
        out_w = int(round(meta.width * target_scale))
        out_h = int(round(meta.height * target_scale))
        self.logger.info(f"Target output: {out_w}x{out_h} (scale factor {target_scale}x)")

        # Prepare upscale function
        if self.custom_upscale_fn is not None:
            upscale_tile = self.custom_upscale_fn
        elif self.engine is not None:
            upscale_tile = self.engine.upscale
        else:
            self.logger.warning(
                f"Không có inference engine cho model '{self.config.model.path}'. "
                "Tự động chuyển sang chế độ nội suy Lanczos4 chất lượng cao."
            )
            def lanczos_fallback(tile: np.ndarray) -> np.ndarray:
                th, tw = tile.shape[:2]
                return cv2.resize(tile, (tw * scale, th * scale), interpolation=cv2.INTER_LANCZOS4)
            upscale_tile = lanczos_fallback

        # Setup deduplicator
        dedup = FrameDeduplicator(
            enabled=self.config.frame_saving.enable_dedup,
            threshold=self.config.frame_saving.dedup_threshold,
        )

        # Setup thermal governor
        telemetry_file = Path(self.config.checkpoint.dir) / "telemetry.csv" if self.config.checkpoint.enabled else None
        governor = ThermalGovernor(self.config.thermal, telemetry_path=telemetry_file)

        # Setup checkpoint manager
        job_id = f"job_{input_file.stem}"
        chk_mgr = CheckpointManager(self.config.checkpoint.dir, job_id=job_id)
        state = chk_mgr.load_state() if resume else {"processed_frames": 0, "segments": [], "completed": False}

        start_frame = state["processed_frames"] if resume else 0
        segments: list[str] = state["segments"] if resume else []
        if start_frame > 0:
            self.logger.info(f"Resuming from frame {start_frame} ({len(segments)} segments completed).")

        reader = VideoReader(input_file)
        reader.start()

        # Handle SIGINT (Ctrl+C) cleanly if running in main thread
        is_main_thread = (threading.current_thread() is threading.main_thread())
        old_sigint = None
        if is_main_thread:
            try:
                def handle_sigint(sig, frame):
                    self.logger.warning("Cancellation requested (Ctrl+C). Finishing current frame and closing cleanly...")
                    self.cancel()

                old_sigint = signal.signal(signal.SIGINT, handle_sigint)
            except (ValueError, AttributeError):
                old_sigint = None

        current_writer: Optional[VideoWriter] = None
        current_segment_idx = len(segments) + 1
        frames_in_curr_segment = 0
        interval = self.config.checkpoint.interval_frames if self.config.checkpoint.enabled else meta.total_frames + 1

        start_time = time.time()
        processed_count = 0

        try:
            for frame_idx, frame in enumerate(reader):
                if self._cancelled:
                    self.logger.warning("Pipeline cancelled by user.")
                    break

                while self._paused and not self._cancelled:
                    time.sleep(0.1)

                # Skip frames already processed in previous run
                if frame_idx < start_frame:
                    continue

                # Thermal governor check & throttle
                governor.wait_if_needed()

                # Pre-processing: Edge-preserving Bilateral Filter to wash away compression noise before AI
                input_for_ai = frame
                if hasattr(self.config, "pre_process") and self.config.pre_process.denoise > 0.0:
                    strength = self.config.pre_process.denoise
                    sigma = 15.0 + strength * 45.0
                    denoised = cv2.bilateralFilter(frame, d=5, sigmaColor=sigma, sigmaSpace=sigma)
                    input_for_ai = cv2.addWeighted(denoised, strength, frame, 1.0 - strength, 0) if strength < 1.0 else denoised

                # Check frame deduplication
                is_dup, prev_result = dedup.check_duplicate(input_for_ai)
                if is_dup and prev_result is not None:
                    out_frame = prev_result
                else:
                    # AI inference with tiling and blend
                    out_frame = upscale_with_tiling(
                        input_for_ai,
                        upscale_fn=upscale_tile,
                        scale=scale,
                        target_scale=target_scale,
                        tile_size=self.config.inference.tile_size,
                        tile_overlap=self.config.inference.tile_overlap,
                    )

                    # Post-processing: Recover Original Detail / Sharpen / Film Grain
                    if hasattr(self.config, "post_process"):
                        pp = self.config.post_process
                        # 1. Recover Original Detail (Alpha Blend with Lanczos base to suppress hallucination & restore lip texture)
                        if pp.recover_detail > 0.0:
                            base = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                            out_frame = cv2.addWeighted(
                                out_frame, 1.0 - pp.recover_detail,
                                base, pp.recover_detail,
                                0
                            )
                        # 2. Adaptive Sharpening (Unsharp Mask)
                        if pp.sharpen > 0.0:
                            blurred = cv2.GaussianBlur(out_frame, (0, 0), sigmaX=1.5)
                            out_frame = cv2.addWeighted(
                                out_frame, 1.0 + pp.sharpen,
                                blurred, -pp.sharpen,
                                0
                            )
                        # 3. Film Grain (Suppress doll/plastic look)
                        if pp.film_grain > 0.0:
                            noise = np.random.normal(0, pp.film_grain * 255.0, (out_h, out_w, 1)).astype(np.float32)
                            noisy = out_frame.astype(np.float32) + noise
                            out_frame = np.clip(noisy, 0, 255).astype(np.uint8)

                    dedup.register_result(out_frame)

                # Segment writer initialization
                if current_writer is None:
                    if self.config.checkpoint.enabled:
                        seg_path = chk_mgr.get_segment_path(current_segment_idx)
                        current_writer = VideoWriter(
                            output_path=seg_path,
                            width=out_w,
                            height=out_h,
                            fps=meta.fps_fraction,
                            audio_source_path=None,  # Audio muxed in final concat
                            codec=self.config.encode.codec,
                            preset=self.config.encode.preset,
                            crf_or_cq=self.config.encode.crf_or_cq,
                        )
                    else:
                        current_writer = VideoWriter(
                            output_path=output_file,
                            width=out_w,
                            height=out_h,
                            fps=meta.fps_fraction,
                            audio_source_path=input_file if self.config.encode.copy_audio else None,
                            codec=self.config.encode.codec,
                            preset=self.config.encode.preset,
                            crf_or_cq=self.config.encode.crf_or_cq,
                        )

                current_writer.write_frame(out_frame)
                frames_in_curr_segment += 1
                processed_count += 1

                # Segment rollover for checkpointing
                if self.config.checkpoint.enabled and frames_in_curr_segment >= interval:
                    current_writer.close()
                    seg_file = chk_mgr.get_segment_path(current_segment_idx)
                    segments.append(str(seg_file))
                    chk_mgr.save_state(processed_frames=frame_idx + 1, segments=segments, completed=False)
                    self.logger.info(f"Checkpoint saved: Segment {current_segment_idx} ({frame_idx + 1} frames total)")
                    current_writer = None
                    current_segment_idx += 1
                    frames_in_curr_segment = 0

                # Progress reporting
                elapsed = time.time() - start_time
                fps_stat = processed_count / max(0.001, elapsed)
                remain_frames = max(0, meta.total_frames - (frame_idx + 1))
                eta_sec = remain_frames / max(0.001, fps_stat)

                if self.on_progress:
                    self.on_progress(
                        frame_idx + 1,
                        meta.total_frames,
                        fps_stat,
                        eta_sec,
                        governor.get_current_metrics(),
                        dedup.skip_rate,
                    )
        finally:
            if is_main_thread and old_sigint is not None:
                try:
                    signal.signal(signal.SIGINT, old_sigint)
                except (ValueError, AttributeError):
                    pass
            reader.close()
            if current_writer is not None:
                current_writer.close()
                if self.config.checkpoint.enabled and frames_in_curr_segment > 0:
                    seg_file = chk_mgr.get_segment_path(current_segment_idx)
                    segments.append(str(seg_file))
                    chk_mgr.save_state(
                        processed_frames=start_frame + processed_count,
                        segments=segments,
                        completed=(start_frame + processed_count >= meta.total_frames),
                    )

            governor.close()

        # If job finished, concat segments and remux audio
        if not self._cancelled and (start_frame + processed_count >= meta.total_frames or not self.config.checkpoint.enabled):
            if self.config.checkpoint.enabled and segments:
                self.logger.info(f"Concatenating {len(segments)} segments and muxing audio...")
                chk_mgr.concat_segments(
                    segment_paths=segments,
                    final_output_path=output_file,
                    audio_source=input_file if self.config.encode.copy_audio else None,
                )
                chk_mgr.cleanup()

            self.logger.info(f"Upscaling completed successfully! Output saved to: {output_file}")
            self.logger.info(f"Dedup stats: {dedup.skipped_frames}/{dedup.total_frames} frames skipped ({dedup.skip_rate:.1f}%)")
