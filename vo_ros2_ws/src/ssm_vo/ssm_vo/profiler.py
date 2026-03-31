"""
profiler.py — Background GPU/CPU hardware profiler.

Polls nvidia-smi via pynvml every POLL_INTERVAL_MS milliseconds and writes
timestamped rows to results/gpu_log.csv.
"""

import csv
import os
import threading
import time
from pathlib import Path


POLL_INTERVAL_S = 0.5   # 500 ms
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[4] / 'results' / 'gpu_log.csv'


class HardwareProfiler:
    """
    Background thread that records GPU utilisation and VRAM usage.

    Usage
    -----
    profiler = HardwareProfiler(log_path='results/gpu_log.csv')
    profiler.start()
    # ... run inference ...
    profiler.stop()
    summary = profiler.summary()
    """

    def __init__(self, log_path: str | Path | None = None, gpu_index: int = 0) -> None:
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self.gpu_index = gpu_index
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows: list[dict] = []
        self._pynvml_available = False

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._pynvml_available = True
        except Exception:
            pass  # GPU profiling unavailable — profiler still works, logs zeros

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._rows.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._write_csv()

    def _poll(self) -> dict:
        row = {'timestamp': time.time(), 'gpu_util_%': 0, 'vram_used_mb': 0}
        if self._pynvml_available:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem  = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                row['gpu_util_%']   = util.gpu
                row['vram_used_mb'] = mem.used // (1024 * 1024)
            except Exception:
                pass
        return row

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._rows.append(self._poll())
            time.sleep(POLL_INTERVAL_S)

    def _write_csv(self) -> None:
        if not self._rows:
            return
        with open(self.log_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=['timestamp', 'gpu_util_%', 'vram_used_mb'])
            writer.writeheader()
            writer.writerows(self._rows)

    def summary(self) -> dict:
        """Return mean/peak GPU stats from the polling session."""
        if not self._rows:
            return {'gpu_util_mean_%': 0, 'gpu_util_peak_%': 0,
                    'vram_mean_mb': 0, 'vram_peak_mb': 0}
        utils = [r['gpu_util_%'] for r in self._rows]
        vrams = [r['vram_used_mb'] for r in self._rows]
        return {
            'gpu_util_mean_%': float(sum(utils)) / len(utils),
            'gpu_util_peak_%': max(utils),
            'vram_mean_mb':    float(sum(vrams)) / len(vrams),
            'vram_peak_mb':    max(vrams),
        }
