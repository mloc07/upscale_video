"""Tests for frame deduplication and skipping redundant AI inference."""

import numpy as np
import pytest

from ecoup.dedup import FrameDeduplicator


def test_dedup_identical_frames():
    dedup = FrameDeduplicator(enabled=True, threshold=0.5)

    frame1 = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    upscaled_dummy = np.ones((1440, 2560, 3), dtype=np.uint8) * 100

    # Frame 1: first frame should never be skipped
    is_dup, res = dedup.check_duplicate(frame1)
    assert is_dup is False
    assert res is None
    dedup.register_result(upscaled_dummy)

    # Frame 2: identical to frame 1 -> should be skipped
    frame2 = frame1.copy()
    is_dup2, res2 = dedup.check_duplicate(frame2)
    assert is_dup2 is True
    assert res2 is upscaled_dummy

    # Frame 3: very minor noise within threshold
    frame3 = frame1.copy()
    frame3[0:5, 0:5, :] += 1  # Tiny change
    is_dup3, res3 = dedup.check_duplicate(frame3)
    assert is_dup3 is True
    assert res3 is upscaled_dummy

    assert dedup.total_frames == 3
    assert dedup.skipped_frames == 2
    assert abs(dedup.skip_rate - 66.66) < 1.0


def test_dedup_different_frames():
    dedup = FrameDeduplicator(enabled=True, threshold=0.5)

    frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
    upscaled_dummy1 = np.zeros((1440, 2560, 3), dtype=np.uint8)

    is_dup, res = dedup.check_duplicate(frame1)
    assert is_dup is False
    dedup.register_result(upscaled_dummy1)

    # Frame 2: completely different frame (white vs black)
    frame2 = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    is_dup2, res2 = dedup.check_duplicate(frame2)
    assert is_dup2 is False
    assert res2 is None


def test_dedup_reset():
    dedup = FrameDeduplicator(enabled=True, threshold=0.5)

    frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
    upscaled = np.ones((200, 200, 3), dtype=np.uint8) * 50

    dedup.check_duplicate(frame)
    dedup.register_result(upscaled)

    # Reset (e.g., scene change)
    dedup.reset()

    # Even though frame is identical, after reset it should not be skipped
    is_dup, res = dedup.check_duplicate(frame)
    assert is_dup is False
    assert res is None
