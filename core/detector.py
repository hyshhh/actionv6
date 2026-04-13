"""人体检测器 — 基于 YOLOv8 + ByteTrack 目标跟踪"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from models.schemas import BoundingBox, PersonDetection
from utils.logger import get_logger

logger = get_logger()


class PersonDetector:
    """
    使用 YOLOv8 进行人体检测，支持 ByteTrack 目标跟踪。

    支持：
    - 自动下载预训练模型
    - CPU / GPU 推理
    - 可配置置信度阈值
    - 可配置检测分辨率（降低推理时间）
    - ByteTrack 目标跟踪，为每个检测目标分配稳定的 track_id
    - YOLO 隔帧推理（yolo_skip_frames），节省计算资源
    - YOLO 推理 FPS 统计
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "cpu",
        class_ids: list[int] | int = 0,
        detect_width: int = 0,
        detect_height: int = 0,
        tracker_enabled: bool = True,
        tracker_type: str = "bytetrack",
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        nms_iou: float = 0.5,
        with_reid: bool = False,
        yolo_skip_frames: int = 0,
    ):
        """
        Args:
            model_path: YOLOv8 模型路径（首次运行自动下载）
            confidence: 最低置信度阈值
            device: 推理设备 ("cpu" / "cuda:0")
            class_ids: 要检测的类别 ID 列表（如 [0,1,2]）或单个 int，会统一当作"人"处理
            detect_width: 检测推理宽度（0=保持原始分辨率）
            detect_height: 检测推理高度（0=保持原始分辨率）
            tracker_enabled: 是否启用目标跟踪
            tracker_type: 跟踪器类型 ("bytetrack" / "botsort")
            track_high_thresh: 高置信度跟踪阈值
            track_low_thresh: 低置信度跟踪阈值（ByteTrack 二次匹配）
            match_thresh: 匹配阈值
            track_buffer: 跟踪丢失后保留帧数
            nms_iou: NMS IoU 阈值（合并重叠框的严格程度）
            with_reid: BoT-SORT 是否启用 ReID 外观特征匹配
            yolo_skip_frames: YOLO 隔帧推理间隔。0=每帧推理，N=每N帧推理一次（功能1）
        """
        self.confidence = confidence
        self.device = device
        # 统一为列表，支持多个类别 ID 都当作"人"
        self.class_ids = [class_ids] if isinstance(class_ids, int) else list(class_ids)
        self.nms_iou = nms_iou
        self.detect_width = detect_width
        self.detect_height = detect_height
        self.tracker_enabled = tracker_enabled
        self.tracker_type = tracker_type
        self.yolo_skip_frames = max(0, yolo_skip_frames)

        # 功能1：隔帧推理 — 缓存上一次的检测结果
        self._cached_detections: list[PersonDetection] = []
        self._last_detection_frame_index: int = -1

        # 功能3：FPS 统计
        self._inference_times: list[float] = []  # 每次实际推理耗时列表
        self._total_inference_count: int = 0      # 实际推理总次数

        logger.info(f"加载 YOLOv8 模型: {model_path} (device={device})")
        self.model = YOLO(model_path)

        # 构建 tracker 配置文件
        self.tracker_config = self._build_tracker_config(
            tracker_type, track_high_thresh, track_low_thresh,
            match_thresh, track_buffer, with_reid,
        )

        mode = f"跟踪({tracker_type})" if tracker_enabled else "仅检测"
        skip_info = f"每帧推理" if self.yolo_skip_frames == 0 else f"每{self.yolo_skip_frames + 1}帧推理1次"
        logger.info(
            f"模型加载完成, 模式={mode}, "
            f"检测分辨率: {'原始' if detect_width == 0 else f'{detect_width}x{detect_height}'}, "
            f"YOLO推理: {skip_info}"
        )

    @staticmethod
    def _build_tracker_config(
        tracker_type: str,
        track_high_thresh: float,
        track_low_thresh: float,
        match_thresh: float,
        track_buffer: int,
        with_reid: bool = False,
    ) -> str:
        """生成 tracker YAML 配置文件并返回路径"""
        import tempfile
        import os

        if tracker_type == "bytetrack":
            content = f"""\
tracker_type: bytetrack
track_high_thresh: {track_high_thresh}
track_low_thresh: {track_low_thresh}
new_track_thresh: {track_low_thresh}
match_thresh: {match_thresh}
track_buffer: {track_buffer}
fuse_score: true
"""
        elif tracker_type == "botsort":
            content = f"""\
tracker_type: botsort
track_high_thresh: {track_high_thresh}
track_low_thresh: {track_low_thresh}
new_track_thresh: {track_low_thresh}
match_thresh: {match_thresh}
track_buffer: {track_buffer}
fuse_score: true
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: {str(with_reid).lower()}
model: auto
"""
        else:
            raise ValueError(f"不支持的跟踪器类型: {tracker_type}")

        # 写入临时文件
        config_path = os.path.join(tempfile.gettempdir(), f"tracker_{tracker_type}.yaml")
        with open(config_path, "w") as f:
            f.write(content)

        return config_path

    def _should_skip_detection(self, frame_index: int) -> bool:
        """
        判断当前帧是否应跳过 YOLO 推理（功能1）。

        规则：
        - yolo_skip_frames == 0：永远不跳过（每帧推理）
        - yolo_skip_frames > 0：距离上次实际推理不足 skip+1 帧时跳过
        """
        if self.yolo_skip_frames <= 0:
            return False
        if self._last_detection_frame_index < 0:
            return False  # 第一次必须推理
        return (frame_index - self._last_detection_frame_index) <= self.yolo_skip_frames

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[PersonDetection]:
        """
        在单帧图像中检测人体。

        如果启用跟踪，使用 model.track() 返回带 track_id 的结果；
        否则使用 model.predict() 纯检测。

        功能1：如果 yolo_skip_frames > 0，非推理帧直接返回上次缓存结果。

        Args:
            frame: BGR 格式的帧图像
            frame_index: 帧序号（用于追踪）

        Returns:
            PersonDetection 列表（含 track_id），按置信度降序排列
        """
        # 功能1：检查是否跳过本帧推理
        if self._should_skip_detection(frame_index):
            return self._cached_detections

        infer_start = time.time()
        orig_h, orig_w = frame.shape[:2]

        # 缩放到检测分辨率
        if self.detect_width > 0 and self.detect_height > 0:
            infer_frame = cv2.resize(
                frame, (self.detect_width, self.detect_height),
                interpolation=cv2.INTER_LINEAR,
            )
            scale_x = orig_w / self.detect_width
            scale_y = orig_h / self.detect_height
        else:
            infer_frame = frame
            scale_x = 1.0
            scale_y = 1.0

        # 选择检测或跟踪模式
        if self.tracker_enabled:
            results = self.model.track(
                infer_frame,
                device=self.device,
                classes=self.class_ids,
                conf=self.confidence,
                iou=self.nms_iou,
                max_det=50,
                tracker=self.tracker_config,
                persist=True,
                verbose=False,
            )
        else:
            results = self.model(
                infer_frame,
                device=self.device,
                classes=self.class_ids,
                conf=self.confidence,
                iou=self.nms_iou,
                max_det=50,
                verbose=False,
            )

        detections: list[PersonDetection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # 提取坐标和置信度
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                # 提取 track_id（跟踪模式下可用）
                track_id = None
                if self.tracker_enabled and box.id is not None:
                    track_id = int(box.id[0].cpu().numpy())

                # 将坐标映射回原始分辨率
                x1 = float(xyxy[0]) * scale_x
                y1 = float(xyxy[1]) * scale_y
                x2 = float(xyxy[2]) * scale_x
                y2 = float(xyxy[3]) * scale_y

                bbox = BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                )

                # 过滤太小的框（可能是噪声）
                if bbox.width < 20 or bbox.height < 20:
                    continue

                detections.append(
                    PersonDetection(
                        frame_index=frame_index,
                        timestamp=time.time(),
                        bbox=bbox,
                        track_id=track_id,
                    )
                )

        # 按置信度降序
        detections.sort(key=lambda d: d.bbox.confidence, reverse=True)

        # 功能1：更新缓存
        self._cached_detections = detections
        self._last_detection_frame_index = frame_index

        # 功能3：记录推理耗时
        infer_elapsed = time.time() - infer_start
        self._inference_times.append(infer_elapsed)
        self._total_inference_count += 1

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[PersonDetection]]:
        """批量检测多帧"""
        return [self.detect(frame, i) for i, frame in enumerate(frames)]

    @property
    def fps(self) -> float:
        """功能3：返回 YOLO 推理平均 FPS"""
        if not self._inference_times:
            return 0.0
        avg_time = sum(self._inference_times) / len(self._inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_fps_stats(self) -> dict:
        """功能3：返回 YOLO 推理详细统计"""
        if not self._inference_times:
            return {"fps": 0.0, "avg_ms": 0.0, "count": 0, "min_ms": 0.0, "max_ms": 0.0}
        times_ms = [t * 1000 for t in self._inference_times]
        avg_ms = sum(times_ms) / len(times_ms)
        return {
            "fps": round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0,
            "avg_ms": round(avg_ms, 2),
            "count": self._total_inference_count,
            "min_ms": round(min(times_ms), 2),
            "max_ms": round(max(times_ms), 2),
        }
