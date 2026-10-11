"""Tiling and overlap blending algorithms for low-VRAM image upscaling."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Callable, Optional
from ecoup.engine.base import UpscaleEngine


def create_blend_mask(height: int, width: int, overlap: int) -> np.ndarray:
    """Create a 2D trapezoidal feathering weight mask for seamless tile blending."""
    if overlap <= 0:
        return np.ones((height, width), dtype=np.float32)

    # 1D ramp for height
    h_ramp = np.ones(height, dtype=np.float32)
    h_ramp[:overlap] = np.linspace(0.01, 1.0, overlap, dtype=np.float32)
    h_ramp[-overlap:] = np.linspace(1.0, 0.01, overlap, dtype=np.float32)

    # 1D ramp for width
    w_ramp = np.ones(width, dtype=np.float32)
    w_ramp[:overlap] = np.linspace(0.01, 1.0, overlap, dtype=np.float32)
    w_ramp[-overlap:] = np.linspace(1.0, 0.01, overlap, dtype=np.float32)

    # 2D outer product
    mask = np.outer(h_ramp, w_ramp)
    return mask.astype(np.float32)


def upscale_with_tiling(
    image: np.ndarray,
    upscale_fn: Callable[[np.ndarray], np.ndarray],
    scale: int = 4,
    target_scale: Optional[float] = None,
    tile_size: int = 256,
    tile_overlap: int = 16,
) -> np.ndarray:
    """Upscale an RGB image with tiling and linear overlap blending to avoid seams.
    
    Args:
        image: Input RGB image (H, W, 3) in uint8 [0, 255].
        upscale_fn: Callable that accepts a tile (H_tile, W_tile, 3) and returns (H_tile * scale, W_tile * scale, 3).
        scale: Model's native scale factor.
        target_scale: Desired final output scale. If different from scale, Lanczos downscale/upscale is applied.
        tile_size: Tile size in input pixels (0 = no tiling).
        tile_overlap: Overlap border in input pixels.
        
    Returns:
        np.ndarray: Upscaled image in uint8 [0, 255].
    """
    h, w, c = image.shape
    if target_scale is None:
        target_scale = float(scale)

    # If tile_size is 0 or image is smaller than tile_size, run whole image directly
    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        out = upscale_fn(image)
        if target_scale != float(scale):
            target_w = int(round(w * target_scale))
            target_h = int(round(h * target_scale))
            out = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return out

    out_h = h * scale
    out_w = w * scale
    out_tile_size = tile_size * scale
    out_overlap = tile_overlap * scale
    stride = tile_size - tile_overlap

    # Y and X grid coordinates
    y_starts = list(range(0, h, stride))
    x_starts = list(range(0, w, stride))

    accum = np.zeros((out_h, out_w, c), dtype=np.float32)
    weights = np.zeros((out_h, out_w), dtype=np.float32)

    for y in y_starts:
        for x in x_starts:
            # Shift window back if near boundary so tile is always within image
            y1 = max(0, min(y, h - tile_size))
            y2 = min(h, y1 + tile_size)
            x1 = max(0, min(x, w - tile_size))
            x2 = min(w, x1 + tile_size)

            tile_in = image[y1:y2, x1:x2, :]
            tile_h, tile_w = tile_in.shape[:2]

            tile_out = upscale_fn(tile_in).astype(np.float32)
            
            # Create blending weight mask for this tile
            actual_out_h, actual_out_w = tile_out.shape[:2]
            mask = create_blend_mask(actual_out_h, actual_out_w, out_overlap)

            # Boundaries in output coordinates
            oy1 = y1 * scale
            oy2 = oy1 + actual_out_h
            ox1 = x1 * scale
            ox2 = ox1 + actual_out_w

            accum[oy1:oy2, ox1:ox2, :] += tile_out * mask[:, :, np.newaxis]
            weights[oy1:oy2, ox1:ox2] += mask

    # Normalize blended result
    weights = np.maximum(weights, 1e-6)
    out = accum / weights[:, :, np.newaxis]
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)

    # Resize to target scale if target_scale differs from model scale
    if target_scale != float(scale):
        target_w = int(round(w * target_scale))
        target_h = int(round(h * target_scale))
        out = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    return out
