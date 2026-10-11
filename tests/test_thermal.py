"""Tests for ThermalGovernor with simulated temperature variations."""

import time
from pathlib import Path
import pytest

from ecoup.config import ThermalConfig
from ecoup.thermal import GpuMetrics, ThermalGovernor


def test_thermal_governor_below_soft_limit(tmp_path: Path):
    """Below soft temp limit, no sleep throttle should be applied."""
    cfg = ThermalConfig(
        enabled=True,
        temp_soft_c=70,
        temp_hard_c=80,
        temp_resume_c=65,
        throttle_sleep_ms_min=0,
        throttle_sleep_ms_max=500,
        poll_interval_sec=0.1,
    )

    metrics = GpuMetrics(temp_c=55, power_w=20.0, util_pct=50, vram_used_mb=1000, vram_total_mb=4000)
    gov = ThermalGovernor(cfg, telemetry_path=tmp_path / "telemetry.csv", custom_metrics_reader=lambda: metrics)

    try:
        time.sleep(0.15)
        slept_ms = gov.wait_if_needed()
        assert slept_ms == 0.0
    finally:
        gov.close()


def test_thermal_governor_soft_throttle(tmp_path: Path):
    """Between soft and hard temp limit, dynamic sleep throttle is applied."""
    cfg = ThermalConfig(
        enabled=True,
        temp_soft_c=70,
        temp_hard_c=80,
        temp_resume_c=65,
        throttle_sleep_ms_min=0,
        throttle_sleep_ms_max=100,  # 100ms max for fast test
        poll_interval_sec=0.05,
    )

    # 75°C is exactly midway between 70 and 80 -> should sleep ~50ms
    metrics = GpuMetrics(temp_c=75, power_w=40.0, util_pct=60, vram_used_mb=1500, vram_total_mb=4000)
    gov = ThermalGovernor(cfg, telemetry_path=tmp_path / "telemetry.csv", custom_metrics_reader=lambda: metrics)

    try:
        time.sleep(0.1)
        start = time.perf_counter()
        slept_ms = gov.wait_if_needed()
        elapsed = (time.perf_counter() - start) * 1000.0

        assert 45.0 <= slept_ms <= 55.0
        assert elapsed >= 40.0
    finally:
        gov.close()


def test_thermal_governor_telemetry_csv(tmp_path: Path):
    """Verify telemetry CSV is populated with header and entries."""
    telemetry_file = tmp_path / "telemetry.csv"
    cfg = ThermalConfig(
        enabled=True,
        poll_interval_sec=0.05,
    )

    metrics = GpuMetrics(temp_c=62, power_w=30.0, util_pct=70, vram_used_mb=2000, vram_total_mb=4000)
    gov = ThermalGovernor(cfg, telemetry_path=telemetry_file, custom_metrics_reader=lambda: metrics)

    try:
        time.sleep(0.2)
    finally:
        gov.close()

    assert telemetry_file.exists()
    content = telemetry_file.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert len(lines) >= 2  # header + at least 1 row
    assert "timestamp,temp_c,power_w" in lines[0]
    assert "62" in lines[1]
