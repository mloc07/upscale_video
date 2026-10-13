"""Tests for AdaIN color correction and luminance preservation."""

import numpy as np
from ecoup.color_fix import adain_color_fix


def test_adain_color_fix_restores_luminance():
    # Original image with mean ~120 and std ~40
    np.random.seed(42)
    source = np.random.randint(40, 200, (120, 160, 3), dtype=np.uint8)
    
    # Simulate diffusion VAE darkening: 25% darker + crushed shadows
    target = np.clip(source * 0.75 - 10, 0, 255).astype(np.uint8)

    assert target.mean() < source.mean() - 25

    # Run AdaIN color fix
    fixed = adain_color_fix(target, source)

    # Restored mean should be within 1.0 of source
    assert abs(fixed.mean() - source.mean()) < 1.0
    # Restored std should be within 1.0 of source
    assert abs(fixed.std() - source.std()) < 1.0


def test_adain_color_fix_different_resolutions():
    # Source is 64x64, target is upscaled 256x256
    np.random.seed(123)
    source = np.random.randint(60, 180, (64, 64, 3), dtype=np.uint8)
    target = np.random.randint(20, 100, (256, 256, 3), dtype=np.uint8)

    fixed = adain_color_fix(target, source)
    assert fixed.shape == (256, 256, 3)
    assert abs(fixed.mean() - source.mean()) < 1.0


def test_adain_color_fix_idempotent():
    np.random.seed(999)
    source = np.random.randint(50, 210, (100, 100, 3), dtype=np.uint8)
    target = np.clip(source * 0.8, 0, 255).astype(np.uint8)

    fixed_once = adain_color_fix(target, source)
    fixed_twice = adain_color_fix(fixed_once, source)

    np.testing.assert_allclose(fixed_once, fixed_twice, atol=1.0)
