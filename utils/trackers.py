"""性能统计器集合 — Qwen 吞吐/延迟、帧率、码流速度、端到端耗时"""

from __future__ import annotations

import threading
import time

import numpy as np


class QwenFpsTracker:
    """
    Qwen 推理速度统计。

    统计两个维度：
    - latency（延迟）: 单次 API 请求的平均耗时
    - throughput（吞吐）: 多线程并发下，每秒实际完成的请求数（滑动窗口 10 秒）
    """

    def __init__(self, window_seconds: float = 60.0):
        self._lock = threading.Lock()
        self._inference_times: list[float] = []
        self._total_count: int = 0
        self._window_seconds = window_seconds
        self._completion_timestamps: list[float] = []
        self._latency_window_max: int = 200

    def record(self, elapsed: float):
        now = time.time()
        with self._lock:
            self._inference_times.append(elapsed)
            self._total_count += 1
            self._completion_timestamps.append(now)
            if len(self._inference_times) > self._latency_window_max:
                self._inference_times = self._inference_times[self._latency_window_max // 2:]

    def _cleanup_window(self):
        """清理窗口外的旧记录（调用方需持有 _lock）"""
        if not self._completion_timestamps:
            return
        cutoff = time.time() - self._window_seconds
        idx = 0
        while idx < len(self._completion_timestamps) and self._completion_timestamps[idx] < cutoff:
            idx += 1
        if idx > 0:
            self._completion_timestamps = self._completion_timestamps[idx:]

    @property
    def throughput(self) -> float:
        """实际吞吐量：滑动窗口内每秒完成的请求数"""
        with self._lock:
            self._cleanup_window()
            if len(self._completion_timestamps) < 2:
                return 0.0
            span = self._completion_timestamps[-1] - self._completion_timestamps[0]
            if span <= 0:
                return 0.0
            return (len(self._completion_timestamps) - 1) / span

    def get_stats(self) -> dict:
        with self._lock:
            if not self._inference_times:
                return {
                    "throughput": 0.0,
                    "avg_ms": 0.0, "count": 0,
                    "min_ms": 0.0, "max_ms": 0.0,
                }
            window = self._inference_times[-50:]
            times_ms = [t * 1000 for t in window]
            avg_ms = sum(times_ms) / len(times_ms)
            self._cleanup_window()
            if len(self._completion_timestamps) >= 2:
                span = self._completion_timestamps[-1] - self._completion_timestamps[0]
                tp = round((len(self._completion_timestamps) - 1) / span, 2) if span > 0 else 0.0
            else:
                tp = 0.0
            return {
                "throughput": tp,
                "avg_ms": round(avg_ms, 2),
                "count": self._total_count,
                "min_ms": round(min(times_ms), 2),
                "max_ms": round(max(times_ms), 2),
            }


class FrameRateTracker:
    """
    实际画面处理速率统计。

    统计主线程实际处理帧的速度：每秒有多少帧从 YOLO 流过。
    """

    def __init__(self, window_seconds: float = 10.0):
        self._window_seconds = window_seconds
        self._timestamps: list[float] = []
        self._max_timestamps: int = 2000

    def tick(self):
        """每处理一帧调用一次"""
        self._timestamps.append(time.time())
        if len(self._timestamps) > self._max_timestamps:
            self._timestamps = self._timestamps[self._max_timestamps // 2:]

    def _cleanup(self):
        if not self._timestamps:
            return
        cutoff = time.time() - self._window_seconds
        idx = 0
        while idx < len(self._timestamps) and self._timestamps[idx] < cutoff:
            idx += 1
        if idx > 0:
            self._timestamps = self._timestamps[idx:]

    @property
    def frame_rate(self) -> float:
        self._cleanup()
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span

    def get_stats(self) -> dict:
        return {
            "frame_rate": round(self.frame_rate, 1),
            "count": len(self._timestamps),
        }


class TotalFrameTracker:
    """
    总单帧推理耗时统计。

    统计从帧进入流水线到该帧所有处理完成的端到端耗时。
    """

    def __init__(self, window_seconds: float = 10.0):
        self._window_seconds = window_seconds
        self._records: list[tuple[float, float]] = []
        self._total_count: int = 0

    def record(self, elapsed: float):
        self._records.append((time.time(), elapsed))
        self._total_count += 1
        if len(self._records) > 200:
            self._records = self._records[100:]

    def _cleanup(self):
        if not self._records:
            return
        cutoff = time.time() - self._window_seconds
        idx = 0
        while idx < len(self._records) and self._records[idx][0] < cutoff:
            idx += 1
        if idx > 0:
            self._records = self._records[idx:]

    def get_stats(self) -> dict:
        self._cleanup()
        if not self._records:
            return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "fps": 0.0, "count": 0}
        times_ms = [t * 1000 for _, t in self._records]
        avg_ms = sum(times_ms) / len(times_ms)
        return {
            "avg_ms": round(avg_ms, 2),
            "min_ms": round(min(times_ms), 2),
            "max_ms": round(max(times_ms), 2),
            "fps": round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0,
            "count": self._total_count,
        }


class BitrateTracker:
    """
    视频码流速度统计。

    统计原始帧数据吞吐量 (MB/s)。
    """

    def __init__(self, window_seconds: float = 10.0):
        self._window_seconds = window_seconds
        self._records: list[tuple[float, int]] = []
        self._total_bytes: int = 0
        self._max_records: int = 2000

    def record(self, frame: np.ndarray):
        now = time.time()
        nbytes = frame.nbytes
        self._records.append((now, nbytes))
        self._total_bytes += nbytes
        if len(self._records) > self._max_records:
            self._records = self._records[self._max_records // 2:]

    def _cleanup(self):
        if not self._records:
            return
        cutoff = time.time() - self._window_seconds
        idx = 0
        while idx < len(self._records) and self._records[idx][0] < cutoff:
            idx += 1
        if idx > 0:
            self._records = self._records[idx:]

    @property
    def mbps(self) -> float:
        self._cleanup()
        if len(self._records) < 2:
            return 0.0
        total_bytes = sum(b for _, b in self._records)
        span = self._records[-1][0] - self._records[0][0]
        if span <= 0:
            return 0.0
        return total_bytes / span / 1_048_576

    def get_stats(self) -> dict:
        self._cleanup()
        frame_count = len(self._records)
        if frame_count < 2:
            return {"mbps": 0.0, "frames": frame_count, "total_mb": round(self._total_bytes / 1_048_576, 2)}
        total_bytes = sum(b for _, b in self._records)
        span = self._records[-1][0] - self._records[0][0]
        mbps = round(total_bytes / span / 1_048_576, 2) if span > 0 else 0.0
        avg_bytes = total_bytes / frame_count
        return {
            "mbps": mbps,
            "frames": frame_count,
            "avg_frame_kb": round(avg_bytes / 1024, 1),
            "total_mb": round(self._total_bytes / 1_048_576, 2),
        }
