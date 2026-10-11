"""Thermal Governor for GPU temperature, power, and utilization control."""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ecoup.config import ThermalConfig


@dataclass
class GpuMetrics:
    """Snapshot of GPU telemetry."""
    temp_c: int = 0
    power_w: float = 0.0
    util_pct: int = 0
    vram_used_mb: int = 0
    vram_total_mb: int = 0
    timestamp: float = 0.0


class ThermalGovernor:
    """Monitors GPU metrics and throttles processing to prevent thermal throttling or overheating."""

    def __init__(
        self,
        config: ThermalConfig,
        telemetry_path: Optional[str | Path] = None,
        custom_metrics_reader: Optional[Callable[[], Optional[GpuMetrics]]] = None,
    ):
        self.config = config
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        self.custom_metrics_reader = custom_metrics_reader

        self.enabled = config.enabled
        self.current_metrics = GpuMetrics()
        self.last_throttle_ms = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._nvml_initialized = False
        self._nvml_handle = None

        if self.enabled:
            self._init_nvml()
            self._init_telemetry()
            self.start()

    def _init_nvml(self) -> None:
        """Initialize NVML if no custom reader is provided."""
        if self.custom_metrics_reader is not None:
            return

        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_initialized = True
            else:
                self.enabled = False
        except Exception:
            self.enabled = False

    def _init_telemetry(self) -> None:
        """Initialize telemetry CSV file and write header if needed."""
        if not self.telemetry_path:
            return

        os.makedirs(self.telemetry_path.parent, exist_ok=True)
        if not self.telemetry_path.exists():
            with open(self.telemetry_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "temp_c", "power_w", "util_pct", "vram_used_mb", "throttle_ms"])

    def _read_metrics(self) -> Optional[GpuMetrics]:
        """Read metrics from custom reader or NVML."""
        if self.custom_metrics_reader is not None:
            return self.custom_metrics_reader()

        if not self._nvml_initialized or self._nvml_handle is None:
            return None

        try:
            import pynvml  # type: ignore
            temp = pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            try:
                pwr = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
            except Exception:
                pwr = 0.0

            return GpuMetrics(
                temp_c=int(temp),
                power_w=float(pwr),
                util_pct=int(util.gpu),
                vram_used_mb=int(mem.used // (1024 * 1024)),
                vram_total_mb=int(mem.total // (1024 * 1024)),
                timestamp=time.time(),
            )
        except Exception:
            return None

    def _log_telemetry(self, metrics: GpuMetrics, throttle_ms: float) -> None:
        """Append metric sample to telemetry CSV."""
        if not self.telemetry_path:
            return
        try:
            with open(self.telemetry_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{metrics.timestamp:.2f}",
                    metrics.temp_c,
                    f"{metrics.power_w:.1f}",
                    metrics.util_pct,
                    metrics.vram_used_mb,
                    f"{throttle_ms:.1f}",
                ])
        except Exception:
            pass

    def _monitor_loop(self) -> None:
        """Background thread polling GPU status."""
        while self._running:
            m = self._read_metrics()
            if m is not None:
                with self._lock:
                    self.current_metrics = m
                self._log_telemetry(m, self.last_throttle_ms)
            time.sleep(self.config.poll_interval_sec)

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def get_current_metrics(self) -> GpuMetrics:
        """Return the most recent GPU metrics snapshot."""
        with self._lock:
            return self.current_metrics

    def wait_if_needed(self) -> float:
        """Check temperature and throttle or pause if necessary.
        
        Returns:
            float: Number of milliseconds slept for throttling.
        """
        if not self.enabled:
            return 0.0

        metrics = self.get_current_metrics()
        temp = metrics.temp_c

        # Hard limit reached: pause completely until cooled down
        if temp >= self.config.temp_hard_c:
            while temp > self.config.temp_resume_c and self._running:
                time.sleep(self.config.poll_interval_sec)
                metrics = self.get_current_metrics()
                temp = metrics.temp_c
            return 0.0

        # Soft limit reached: dynamic sleep between frames
        sleep_ms = 0.0
        if temp >= self.config.temp_soft_c:
            temp_range = max(1, self.config.temp_hard_c - self.config.temp_soft_c)
            ratio = (temp - self.config.temp_soft_c) / temp_range
            ratio = min(1.0, max(0.0, ratio))
            sleep_ms = self.config.throttle_sleep_ms_min + ratio * (
                self.config.throttle_sleep_ms_max - self.config.throttle_sleep_ms_min
            )

        # High utilization penalty
        if metrics.util_pct > self.config.max_gpu_util_pct:
            sleep_ms += 15.0

        self.last_throttle_ms = sleep_ms
        if sleep_ms > 0.0:
            time.sleep(sleep_ms / 1000.0)

        return sleep_ms

    def close(self) -> None:
        """Stop background thread and shutdown NVML."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._nvml_initialized:
            try:
                import pynvml  # type: ignore
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def __enter__(self) -> ThermalGovernor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
