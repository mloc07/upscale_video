"""Tests for image tiling, overlap blending, and seam prevention."""

import cv2
import numpy as np
import pytest

from ecoup.tiling import create_blend_mask, upscale_with_tiling


def test_create_blend_mask():
    mask = create_blend_mask(height=100, width=100, overlap=10)
    assert mask.shape == (100, 100)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    # Center should be 1.0
    assert mask[50, 50] == 1.0
    # Corners should be small
    assert mask[0, 0] < 0.1


def mock_upscale_2x(img: np.ndarray) -> np.ndarray:
    """Mock upscale function using bicubic 2x."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    return 20.0 * np.log10(max_pixel / np.sqrt(mse))


def test_tiling_vs_nontiling_psnr():
    """Verify that tiled upscaling with blend achieves > 40 dB PSNR vs non-tiled."""
    np.random.seed(42)
    # Create synthetic test image (e.g. 360 x 480) with gradients and noise
    h, w = 360, 480
    y, x = np.ogrid[:h, :w]
    gradient = ((x / w) * 128 + (y / h) * 127).astype(np.uint8)
    noise = np.random.randint(0, 20, (h, w), dtype=np.uint8)
    base = np.clip(gradient + noise, 0, 255)
    img = np.stack([base, np.roll(base, 20, axis=1), np.roll(base, 40, axis=0)], axis=2)

    # 1. Non-tiled upscale
    full_out = upscale_with_tiling(
        img,
        upscale_fn=mock_upscale_2x,
        scale=2,
        target_scale=2,
        tile_size=0,
    )
    assert full_out.shape == (h * 2, w * 2, 3)

    # 2. Tiled upscale with tile_size=128, overlap=16
    tiled_out = upscale_with_tiling(
        img,
        upscale_fn=mock_upscale_2x,
        scale=2,
        target_scale=2,
        tile_size=128,
        tile_overlap=16,
    )
    assert tiled_out.shape == (h * 2, w * 2, 3)

    # 3. PSNR comparison
    psnr = calculate_psnr(full_out, tiled_out)
    assert psnr > 40.0, f"PSNR between tiled and non-tiled is {psnr:.2f} dB, expected > 40 dB"


def test_tiling_with_target_scale_resize():
    """Verify target_scale resizing when target_scale differs from model scale."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Model scale is 4, but target_scale is 2
    out = upscale_with_tiling(
        img,
        upscale_fn=lambda x: cv2.resize(x, (x.shape[1] * 4, x.shape[0] * 4)),
        scale=4,
        target_scale=2.0,
        tile_size=64,
        tile_overlap=8,
    )
    assert out.shape == (200, 200, 3)
