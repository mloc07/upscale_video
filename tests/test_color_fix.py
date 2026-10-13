"""Tests for Wavelet color correction, luminance preservation, and flicker-free video stability."""

import numpy as np
from ecoup.color_fix import wavelet_color_fix, adain_color_fix


def test_wavelet_color_fix_restores_luminance():
    # Original image with mean ~120 and std ~40
    np.random.seed(42)
    source = np.random.randint(40, 200, (120, 160, 3), dtype=np.uint8)
    
    # Simulate diffusion VAE darkening: 25% darker + crushed shadows
    target = np.clip(source * 0.75 - 10, 0, 255).astype(np.uint8)

    assert target.mean() < source.mean() - 25

    # Run color fix
    fixed = wavelet_color_fix(target, source)

    # Restored mean should be within 1.0 of source
    assert abs(fixed.mean() - source.mean()) < 1.0
    # Restored std should be within 1.0 of source
    assert abs(fixed.std() - source.std()) < 1.0


def test_wavelet_color_fix_different_resolutions():
    # Source is 64x64, target is upscaled 256x256
    np.random.seed(123)
    source = np.random.randint(60, 180, (64, 64, 3), dtype=np.uint8)
    target = np.random.randint(20, 100, (256, 256, 3), dtype=np.uint8)

    fixed = wavelet_color_fix(target, source)
    assert fixed.shape == (256, 256, 3)
    assert abs(fixed.mean() - source.mean()) < 1.0


def test_wavelet_color_fix_idempotent():
    np.random.seed(999)
    source = np.random.randint(50, 210, (100, 100, 3), dtype=np.uint8)
    target = np.clip(source * 0.8, 0, 255).astype(np.uint8)

    fixed_once = wavelet_color_fix(target, source)
    fixed_twice = wavelet_color_fix(fixed_once, source)

    np.testing.assert_allclose(fixed_once, fixed_twice, atol=1.0)


def test_wavelet_color_fix_temporal_stability():
    """Verify that random diffusion color drift on consecutive frames does not cause flickering."""
    np.random.seed(10)
    # A base video frame
    base_frame = np.random.randint(60, 200, (100, 100, 3), dtype=np.uint8)
    
    # Simulate 2 consecutive frames where the AI diffusion model drifted in color (one greenish, one reddish)
    ai_frame1 = np.clip(base_frame.astype(float) * [0.70, 0.90, 0.70] - 15, 0, 255).astype(np.uint8)
    ai_frame2 = np.clip(base_frame.astype(float) * [0.90, 0.70, 0.90] - 15, 0, 255).astype(np.uint8)

    fixed1 = wavelet_color_fix(ai_frame1, base_frame)
    fixed2 = wavelet_color_fix(ai_frame2, base_frame)

    # Both fixed frames must be anchored to base_frame color: mean difference between them should be < 1.0
    mean_diff = np.abs(fixed1.mean(axis=(0, 1)) - fixed2.mean(axis=(0, 1))).max()
    assert mean_diff < 1.0, f"Expected flicker-free color anchoring, got max channel mean diff {mean_diff}"
