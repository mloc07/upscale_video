"""Speech recognition with faster-whisper (optional dependency)."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from .config import AsrConfig
from .segments import Segment


@dataclass
class AsrResult:
    segments: list[Segment]
    language: Optional[str]
    language_probability: float = 0.0
    duration_s: float = 0.0
    elapsed_s: float = 0.0
    peak_vram_gb: Optional[float] = None


def _load_wav_float32(path):
    """Read a 16-bit PCM mono WAV into a float32 numpy array in [-1, 1]."""
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV.")
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class AsrBackend(Protocol):
    def transcribe(self, wav_path: str, cfg: AsrConfig,
                   on_segment: Optional[Callable[[float], None]] = None,
                   should_cancel: Optional[Callable[[], bool]] = None) -> AsrResult: ...


class FasterWhisperBackend:
    """Loads the model, transcribes, then releases it (GPU memory is freed afterwards)."""

    def transcribe(self, wav_path, cfg, on_segment=None, should_cancel=None) -> AsrResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "faster-whisper is not installed. Install the optional extra: pip install 'ecoup[subtitle]'"
            ) from exc

        device = cfg.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        compute = cfg.compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"

        t0 = time.time()
        model = WhisperModel(cfg.model, device=device, compute_type=compute)
        peak = None
        try:
            kwargs = dict(
                beam_size=cfg.beam_size,
                language=cfg.language,
                vad_filter=cfg.vad_filter,
                condition_on_previous_text=cfg.condition_on_previous_text,
            )
            if cfg.vad_filter:
                kwargs["vad_parameters"] = dict(min_silence_duration_ms=cfg.vad_min_silence_ms)
            # Feed a float32 array (our own 16 kHz mono WAV) so faster-whisper never needs PyAV to decode;
            # this avoids PyAV/faster-whisper version mismatches seen on Colab.
            seg_iter, info = model.transcribe(_load_wav_float32(wav_path), **kwargs)
            segs: list[Segment] = []
            for s in seg_iter:  # lazy generator: honour cancellation between segments
                if should_cancel and should_cancel():
                    break
                segs.append(Segment(start=float(s.start), end=float(s.end), text=s.text.strip(), language=info.language))
                if on_segment:
                    on_segment(float(s.end))
            if device == "cuda":
                try:
                    import torch
                    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                except Exception:
                    peak = None
            return AsrResult(
                segments=segs,
                language=info.language,
                language_probability=float(getattr(info, "language_probability", 0.0)),
                duration_s=float(getattr(info, "duration", 0.0)),
                elapsed_s=time.time() - t0,
                peak_vram_gb=peak,
            )
        finally:
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
