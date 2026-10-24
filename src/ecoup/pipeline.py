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
from ecoup.color_fix import wavelet_color_fix, adain_color_fix
from ecoup.config import AppConfig
from ecoup.dedup import FrameDeduplicator
from ecoup.engine.base import UpscaleEngine, VideoUpscaleEngine
from ecoup.thermal import ThermalGovernor
from ecoup.tiling import upscale_with_tiling
from ecoup.utils import setup_logger, format_duration
from ecoup.video_io import VideoReader, VideoWriter, get_video_metadata


class UpscalePipeline:
    """Coordinates VideoReader, Dedup, ThermalGovernor, AI Inference, and VideoWriter."""

    def __init__(
        self,
        config: AppConfig,
        engine: Optional[UpscaleEngine | VideoUpscaleEngine] = None,
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

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Pre-processing: Edge-preserving Bilateral Filter / NLMeans to wash away compression noise."""
        if not hasattr(self.config, "pre_process"):
            return frame
        pp = self.config.pre_process
        method = getattr(pp, "method", "bilateral").lower()
        if method == "none" or pp.denoise <= 0.0:
            return frame

        strength = pp.denoise
        if method == "nlmeans":
            h_val = max(1.0, strength * 10.0)
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, h=h_val, hColor=h_val, templateWindowSize=7, searchWindowSize=15)
            return cv2.addWeighted(denoised, strength, frame, 1.0 - strength, 0) if strength < 1.0 else denoised
        else:
            sigma = 15.0 + strength * 45.0
            denoised = cv2.bilateralFilter(frame, d=5, sigmaColor=sigma, sigmaSpace=sigma)
            return cv2.addWeighted(denoised, strength, frame, 1.0 - strength, 0) if strength < 1.0 else denoised

    def _postprocess_frame(self, out_frame: np.ndarray, orig_frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        """Post-processing: Color Fix, Recover Original Detail, Sharpening, and Film Grain."""
        if out_frame.shape[1] != out_w or out_frame.shape[0] != out_h:
            out_frame = cv2.resize(out_frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        if hasattr(self.config, "post_process"):
            pp = self.config.post_process
            if getattr(pp, "color_fix", False):
                out_frame = wavelet_color_fix(out_frame, orig_frame)
            if pp.recover_detail > 0.0:
                base = cv2.resize(orig_frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                out_frame = cv2.addWeighted(out_frame, 1.0 - pp.recover_detail, base, pp.recover_detail, 0)
            if pp.sharpen > 0.0:
                blurred = cv2.GaussianBlur(out_frame, (0, 0), sigmaX=1.5)
                out_frame = cv2.addWeighted(out_frame, 1.0 + pp.sharpen, blurred, -pp.sharpen, 0)
            if pp.film_grain > 0.0:
                noise = np.random.normal(0, pp.film_grain * 255.0, (out_h, out_w, 1)).astype(np.float32)
                noisy = out_frame.astype(np.float32) + noise
                out_frame = np.clip(noisy, 0, 255).astype(np.uint8)
        return out_frame

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
        elif self.engine is not None and hasattr(self.engine, "upscale"):
            if hasattr(self.engine, "control_strength") and hasattr(self.config.inference, "control_strength"):
                self.engine.control_strength = self.config.inference.control_strength
            upscale_tile = self.engine.upscale
        elif self.engine is not None and hasattr(self.engine, "upscale_clip"):
            upscale_tile = None
            if hasattr(self.engine, "scale"):
                self.engine.scale = int(target_scale) if float(target_scale).is_integer() else float(target_scale)
            if hasattr(self.engine, "seed") and hasattr(self.config.inference, "seed"):
                self.engine.seed = self.config.inference.seed
            if hasattr(self.engine, "sub_len") and hasattr(self.config.inference, "sub_len"):
                self.engine.sub_len = self.config.inference.sub_len
            if hasattr(self.engine, "model_resolution") and hasattr(self.config.inference, "model_resolution"):
                self.engine.model_resolution = self.config.inference.model_resolution
            if hasattr(self.engine, "memory_mode") and hasattr(self.config.inference, "memory_mode"):
                self.engine.memory_mode = self.config.inference.memory_mode
            if hasattr(self.engine, "precision") and hasattr(self.config.model, "precision"):
                self.engine.precision = self.config.model.precision
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

        # Check if engine is clip-based
        is_clip_engine = isinstance(self.engine, VideoUpscaleEngine) or hasattr(self.engine, "upscale_clip")

        def write_frame_out(out_f: np.ndarray, frame_num: int) -> None:
            nonlocal current_writer, current_segment_idx, frames_in_curr_segment, processed_count
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

            current_writer.write_frame(out_f)
            frames_in_curr_segment += 1
            processed_count += 1

            # Segment rollover for checkpointing
            if self.config.checkpoint.enabled and frames_in_curr_segment >= interval:
                current_writer.close()
                seg_file = chk_mgr.get_segment_path(current_segment_idx)
                segments.append(str(seg_file))
                chk_mgr.save_state(processed_frames=frame_num, segments=segments, completed=False)
                self.logger.info(f"Checkpoint saved: Segment {current_segment_idx} ({frame_num} frames total)")
                current_writer = None
                current_segment_idx += 1
                frames_in_curr_segment = 0

            # Progress reporting
            elapsed = time.time() - start_time
            fps_stat = processed_count / max(0.001, elapsed)
            remain_frames = max(0, meta.total_frames - frame_num)
            eta_sec = remain_frames / max(0.001, fps_stat)

            if self.on_progress:
                self.on_progress(
                    frame_num,
                    meta.total_frames,
                    fps_stat,
                    eta_sec,
                    governor.get_current_metrics(),
                    0.0 if is_clip_engine else dedup.skip_rate,
                )

        try:
            if is_clip_engine:
                clip_len = getattr(self.config.inference, "clip_length", 16)
                clip_ov = getattr(self.config.inference, "clip_overlap", 2)
                step = max(1, clip_len - clip_ov)

                reader_iter = iter(reader)
                # Skip frames already processed in previous run
                for _ in range(start_frame):
                    try:
                        next(reader_iter)
                    except StopIteration:
                        break

                overlap_input_buffer: list[np.ndarray] = []
                prev_tail: list[np.ndarray] = []
                is_eof = False
                chunk_idx = 0

                while not is_eof and not self._cancelled:
                    while self._paused and not self._cancelled:
                        time.sleep(0.1)

                    governor.wait_if_needed()

                    raw_chunk = list(overlap_input_buffer)
                    needed = clip_len - len(raw_chunk)
                    for _ in range(needed):
                        try:
                            f = next(reader_iter)
                            raw_chunk.append(f)
                        except StopIteration:
                            is_eof = True
                            break

                    if not raw_chunk:
                        break

                    # Pre-processing
                    preprocessed_chunk = [self._preprocess_frame(f) for f in raw_chunk]

                    # AI Inference on clip
                    upscaled_chunk = self.engine.upscale_clip(preprocessed_chunk)

                    # Post-processing
                    postprocessed_chunk = [
                        self._postprocess_frame(up_f, orig_f, out_w, out_h)
                        for orig_f, up_f in zip(raw_chunk, upscaled_chunk)
                    ]

                    # Update overlap buffer for next chunk
                    if not is_eof and clip_ov > 0 and len(raw_chunk) >= clip_ov:
                        overlap_input_buffer = raw_chunk[-clip_ov:]
                    else:
                        overlap_input_buffer = []

                    # Output frames with cross-fade
                    if chunk_idx == 0:
                        if is_eof or clip_ov == 0:
                            for f in postprocessed_chunk:
                                write_frame_out(f, start_frame + processed_count + 1)
                        else:
                            for f in postprocessed_chunk[:step]:
                                write_frame_out(f, start_frame + processed_count + 1)
                            prev_tail = postprocessed_chunk[step:]
                    else:
                        curr_head = postprocessed_chunk[:clip_ov]
                        fade_len = min(len(prev_tail), len(curr_head))
                        for k in range(fade_len):
                            alpha = float(k + 1) / float(fade_len + 1)
                            blended = cv2.addWeighted(prev_tail[k], 1.0 - alpha, curr_head[k], alpha, 0)
                            write_frame_out(blended, start_frame + processed_count + 1)

                        if is_eof or clip_ov == 0:
                            for f in postprocessed_chunk[clip_ov:]:
                                write_frame_out(f, start_frame + processed_count + 1)
                            prev_tail = []
                        else:
                            for f in postprocessed_chunk[clip_ov : step]:
                                write_frame_out(f, start_frame + processed_count + 1)
                            prev_tail = postprocessed_chunk[step:]

                    chunk_idx += 1
            else:
                for frame_idx, frame in enumerate(reader):
                    if self._cancelled:
                        self.logger.warning("Pipeline cancelled by user.")
                        break

                    while self._paused and not self._cancelled:
                        time.sleep(0.1)

                    if frame_idx < start_frame:
                        continue

                    governor.wait_if_needed()

                    input_for_ai = self._preprocess_frame(frame)

                    is_dup, prev_result = dedup.check_duplicate(input_for_ai)
                    if is_dup and prev_result is not None:
                        out_frame = prev_result
                    else:
                        out_frame = upscale_with_tiling(
                            input_for_ai,
                            upscale_fn=upscale_tile,
                            scale=scale,
                            target_scale=target_scale,
                            tile_size=self.config.inference.tile_size,
                            tile_overlap=self.config.inference.tile_overlap,
                        )
                        out_frame = self._postprocess_frame(out_frame, frame, out_w, out_h)
                        dedup.register_result(out_frame)

                    write_frame_out(out_frame, frame_idx + 1)
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
                for s in segments:
                    meta_s = get_video_metadata(s)
                    self.logger.info(f"  Segment {Path(s).name}: {meta_s.total_frames} frames")
                chk_mgr.concat_segments(
                    segment_paths=segments,
                    final_output_path=output_file,
                    audio_source=input_file if self.config.encode.copy_audio else None,
                )
                chk_mgr.cleanup()

            self.logger.info(f"Upscaling completed successfully! Output saved to: {output_file}")
            self.logger.info(f"Dedup stats: {dedup.skipped_frames}/{dedup.total_frames} frames skipped ({dedup.skip_rate:.1f}%)")

            # Save sidecar metadata JSON
            metadata_json = output_file.with_suffix(".json")
            try:
                import json
                run_meta = {
                    "input_file": str(input_file),
                    "output_file": str(output_file),
                    "original_width": meta.width,
                    "original_height": meta.height,
                    "target_width": out_w,
                    "target_height": out_h,
                    "fps": float(meta.fps),
                    "total_frames": meta.total_frames,
                    "processed_frames": start_frame + processed_count,
                    "model_name": self.config.model.name,
                    "scale": scale,
                    "target_scale": target_scale,
                    "seed": getattr(self.config.inference, "seed", None),
                    "precision": getattr(self.config.model, "precision", "fp16"),
                    "memory_mode": getattr(self.config.inference, "memory_mode", "auto"),
                    "sub_len": getattr(self.config.inference, "sub_len", 5),
                    "model_resolution": getattr(self.config.inference, "model_resolution", "native"),
                    "elapsed_sec": round(time.time() - start_time, 2),
                }
                with open(metadata_json, "w", encoding="utf-8") as f:
                    json.dump(run_meta, f, indent=2)
                self.logger.info(f"Đã lưu siêu dữ liệu cấu hình vào: {metadata_json}")
            except Exception as e:
                self.logger.warning(f"Không thể lưu metadata JSON: {e}")
