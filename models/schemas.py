"""数据模型定义"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    """行为严重等级"""
    CRITICAL = "critical"   # 危险（如溺水）
    WARNING = "warning"     # 警告（如翻栏杆）
    NORMAL = "normal"       # 正常


@dataclass
class BoundingBox:
    """人体检测框"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.width / 2, self.y1 + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class PersonDetection:
    """单帧人体检测结果"""
    frame_index: int
    timestamp: float
    bbox: BoundingBox
    track_id: Optional[int] = None  # ByteTrack 跟踪 ID（无跟踪时为 None）
    cropped_image: Optional[bytes] = None  # JPEG 编码的裁剪图


@dataclass
class BehaviorResult:
    """行为识别结果"""
    behavior_id: str
    behavior_label: str
    description: str
    severity: Severity
    confidence: float = 0.0  # 模型给出的置信度（如可用）

    def is_alert(self) -> bool:
        """是否需要告警"""
        return self.severity in (Severity.CRITICAL, Severity.WARNING)


@dataclass
class FrameAnalysis:
    """单帧分析的完整结果"""
    frame_index: int
    timestamp: float
    frame_width: int
    frame_height: int
    detections: list[PersonDetection] = field(default_factory=list)
    behaviors: list[BehaviorResult] = field(default_factory=list)
    behavior_dicts: list[dict] = field(default_factory=list)  # 含 person_key 的行为 dict 列表
    processing_time: float = 0.0


@dataclass
class AnalysisReport:
    """完整分析报告"""
    source: str                     # 输入源描述
    start_time: float = 0.0
    end_time: float = 0.0
    total_frames: int = 0
    processed_frames: int = 0
    total_detections: int = 0
    behavior_counts: dict[str, int] = field(default_factory=dict)
    alerts: list[dict] = field(default_factory=list)
    frame_analyses: list[FrameAnalysis] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "source": self.source,
            "duration_seconds": round(self.end_time - self.start_time, 2),
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "total_detections": self.total_detections,
            "behavior_counts": self.behavior_counts,
            "alert_count": len(self.alerts),
        }
