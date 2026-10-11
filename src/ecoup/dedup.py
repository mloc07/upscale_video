"""Frame deduplication and static frame detector to skip redundant AI inference."""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


class FrameDeduplicator:
    """Detects identical or nearly identical frames to reuse previous upscale result."""

    def __init__(
        self,
        enabled: bool = True,
        threshold: float = 0.5,
        thumb_width: int = 160,
    ):
        self.enabled = enabled
        self.threshold = threshold
        self.thumb_width = thumb_width

        self.last_thumb: Optional[np.ndarray] = None
        self.current_thumb: Optional[np.ndarray] = None
        self.last_upscaled: Optional[np.ndarray] = None

        self.total_frames = 0
        self.skipped_frames = 0

    def _compute_thumbnail(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to small grayscale thumbnail for fast comparison."""
        h, w = frame.shape[:2]
        thumb_h = max(1, int(round(h * (self.thumb_width / w))))
        small = cv2.resize(frame, (self.thumb_width, thumb_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        return gray

    def check_duplicate(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Check if current frame is duplicate of previous frame.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (is_duplicate, previous_upscaled_result)
        """
        if not self.enabled:
            self.total_frames += 1
            return False, None

        self.current_thumb = self._compute_thumbnail(frame)

        if self.last_thumb is None or self.last_upscaled is None:
            self.total_frames += 1
            return False, None

        # Mean Squared Error between grayscale thumbnails
        mse = float(np.mean((self.current_thumb.astype(np.float32) - self.last_thumb.astype(np.float32)) ** 2))

        if mse <= self.threshold:
            self.total_frames += 1
            self.skipped_frames += 1
            return True, self.last_upscaled

        self.total_frames += 1
        return False, None

    def register_result(self, upscaled: np.ndarray) -> None:
        """Save the upscaled result and current thumbnail for subsequent comparisons."""
        self.last_thumb = self.current_thumb
        self.last_upscaled = upscaled

    def reset(self) -> None:
        """Reset state (e.g., when a new scene is detected)."""
        self.last_thumb = None
        self.current_thumb = None
        self.last_upscaled = None

    @property
    def skip_rate(self) -> float:
        """Percentage of frames skipped (0.0 to 100.0)."""
        if self.total_frames == 0:
            return 0.0
        return (self.skipped_frames / self.total_frames) * 100.0
