"""Base interface for AI upscale engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class UpscaleEngine(ABC):
    """Abstract base class for upscale engines."""

    @abstractmethod
    def load(self, model_path: str, precision: str = "fp16", provider: str = "auto") -> None:
        """Load model weights and initialize inference session."""
        pass

    @abstractmethod
    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a single RGB image/frame (H, W, 3) in uint8 [0, 255].
        
        Returns:
            np.ndarray: Upscaled RGB image (H * scale, W * scale, 3) in uint8.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release GPU and runtime resources."""
        pass

    def __enter__(self) -> UpscaleEngine:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
