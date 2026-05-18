"""摄像头行为日志管理器 — 线程安全的 JSON 日志"""

from __future__ import annotations

import json
import os
import threading
import time

from utils.logger import get_logger

logger = get_logger()


class CameraBehaviorLog:
    """
    摄像头行为日志管理器（线程安全）。

    功能：
    - 记录每次行为识别结果，包含时间戳和行为信息
    - 定期清理超过保留时长的日志条目
    - 以 JSON 格式持久化存储
    - 批量写入：累积条目后定时/阈值触发落盘，避免每条都写
    """

    def __init__(
        self,
        output_dir: str,
        retention_hours: float = 2.0,
        log_filename: str = "camera_behavior_log.json",
        flush_interval: float = 5.0,
        flush_threshold: int = 20,
    ):
        self.output_dir = output_dir
        self.retention_seconds = retention_hours * 3600
        self.log_path = os.path.join(output_dir, log_filename)
        self._flush_interval = flush_interval
        self._flush_threshold = flush_threshold

        self._lock = threading.Lock()
        self._entries: list[dict] = []
        self._last_flush_time: float = time.time()
        self._load_existing()

    def _load_existing(self):
        """加载已有的日志文件"""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
                logger.info(f"已加载摄像头日志: {len(self._entries)} 条记录")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"加载日志失败，重新开始: {e}")
                self._entries = []

    def add_entry(
        self,
        frame_index: int,
        person_idx: int,
        behavior_id: str,
        behavior_label: str,
        severity: str,
        description: str,
    ):
        now = time.time()
        entry = {
            "timestamp": round(now, 3),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "frame_index": frame_index,
            "person_idx": person_idx,
            "behavior_id": behavior_id,
            "behavior_label": behavior_label,
            "severity": severity,
            "description": description,
        }
        with self._lock:
            self._entries.append(entry)
            elapsed = now - self._last_flush_time
            if len(self._entries) >= self._flush_threshold or elapsed >= self._flush_interval:
                self._flush_unlocked()
                self._last_flush_time = now

    def _cleanup_unlocked(self):
        """清理过期条目（调用方需持有 _lock）"""
        now = time.time()
        cutoff = now - self.retention_seconds
        before_count = len(self._entries)
        self._entries = [e for e in self._entries if e.get("timestamp", 0) >= cutoff]
        removed = before_count - len(self._entries)
        if removed > 0:
            logger.debug(f"摄像头日志清理: 删除 {removed} 条过期记录")

    def _flush_unlocked(self):
        """将内存中的条目写入磁盘（调用方需持有 _lock）"""
        self._cleanup_unlocked()
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存摄像头日志失败: {e}")

    def save(self):
        """手动触发一次落盘"""
        with self._lock:
            self._flush_unlocked()
            self._last_flush_time = time.time()
        logger.info(f"摄像头日志文件已写入: {self.log_path}")

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)
