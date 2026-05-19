"""人体检测器 — 基于 YOLOv8 + SAHI + ByteTrack 目标跟踪"""

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
    使用 YOLOv8 进行人体检测，支持 SAHI 小目标增强 + ByteTrack 目标跟踪。

    支持：
    - 自动下载预训练模型
    - CPU / GPU 推理
    - 可配置置信度阈值
    - 可配置检测分辨率（降低推理时间）
    - SAHI 切片推理（提升小目标检出率）
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
        min_bbox_size: int = 8,
        sahi_enabled: bool = False,
        sahi_batch_enabled: bool = True,
        sahi_slice_width: int = 640,
        sahi_slice_height: int = 640,
        sahi_overlap: float = 0.2,
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
            yolo_skip_frames: YOLO 隔帧推理间隔。0=每帧推理，N=每N帧推理一次
            min_bbox_size: 最小检测框像素（宽或高低于此值的框被过滤）
            sahi_enabled: 是否启用 SAHI 切片推理（提升小目标检出率）
            sahi_batch_enabled: SAHI 是否启用批量推理（更快，默认开启）
            sahi_slice_width: SAHI 切片宽度
            sahi_slice_height: SAHI 切片高度
            sahi_overlap: SAHI 切片重叠比例
        """
        self.confidence = confidence
        self.device = device
        self.class_ids = [class_ids] if isinstance(class_ids, int) else list(class_ids)
        self.nms_iou = nms_iou
        self.detect_width = detect_width
        self.detect_height = detect_height
        self.tracker_enabled = tracker_enabled
        self.tracker_type = tracker_type
        self.yolo_skip_frames = max(0, yolo_skip_frames)
        self.min_bbox_size = min_bbox_size

        # SAHI 配置
        self.sahi_enabled = sahi_enabled
        self.sahi_batch_enabled = sahi_batch_enabled
        self.sahi_slice_width = sahi_slice_width
        self.sahi_slice_height = sahi_slice_height
        self.sahi_overlap = sahi_overlap
        self._sahi_model = None

        # 隔帧推理缓存
        self._cached_detections: list[PersonDetection] = []
        self._last_detection_frame_index: int = -1

        # FPS 统计
        self._inference_times: list[float] = []
        self._inference_window_max: int = 200
        self._total_inference_count: int = 0

        logger.info(f"加载 YOLOv8 模型: {model_path} (device={device})")
        self.model = YOLO(model_path)

        # 初始化 SAHI（延迟加载，首次推理时创建）
        if self.sahi_enabled:
            batch_mode = "批量推理" if sahi_batch_enabled else "逐片推理"
            logger.info(f"SAHI 已启用: slice={sahi_slice_width}x{sahi_slice_height}, overlap={sahi_overlap}, 模式={batch_mode}")

        # 构建 tracker 配置文件
        self.tracker_config = self._build_tracker_config(
            tracker_type, track_high_thresh, track_low_thresh,
            match_thresh, track_buffer, with_reid,
        )

        mode = f"跟踪({tracker_type})" if tracker_enabled else "仅检测"
        if sahi_enabled:
            batch_tag = "批量" if sahi_batch_enabled else "逐片"
            mode += f" + SAHI({batch_tag})"
        skip_info = f"每帧推理" if self.yolo_skip_frames == 0 else f"每{self.yolo_skip_frames + 1}帧推理1次"
        logger.info(
            f"模型加载完成, 模式={mode}, "
            f"检测分辨率: {'原始' if detect_width == 0 else f'{detect_width}x{detect_height}'}, "
            f"YOLO推理: {skip_info}"
        )

    def _init_sahi_model(self):
        """延迟初始化 SAHI 模型（首次推理时创建）"""
        if self._sahi_model is not None:
            return
        try:
            from sahi import AutoDetectionModel
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.model.model_name if hasattr(self.model, 'model_name') else str(self.model),
                confidence_threshold=self.confidence,
                device=self.device,
            )
            logger.info("SAHI 模型初始化完成")
        except ImportError:
            logger.error("sahi 未安装，请运行: pip install sahi")
            self.sahi_enabled = False
        except Exception as e:
            logger.error(f"SAHI 初始化失败: {e}")
            self.sahi_enabled = False

    @staticmethod
    def _build_tracker_config(
        tracker_type: str,
        track_high_thresh: float,
        track_low_thresh: float,
        match_thresh: float,
        track_buffer: int,
        with_reid: bool = False,
    ) -> str:
        """生成 tracker YAML 配置文件并返回路径（全局缓存，避免重复写入）"""
        import tempfile
        import os

        param_hash = f"{tracker_type}_{track_high_thresh}_{track_low_thresh}_{match_thresh}_{track_buffer}_{with_reid}"
        config_path = os.path.join(tempfile.gettempdir(), f"tracker_{param_hash}.yaml")

        if os.path.exists(config_path):
            return config_path

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

        with open(config_path, "w") as f:
            f.write(content)

        return config_path

    def _should_skip_detection(self, frame_index: int) -> bool:
        """判断当前帧是否应跳过推理"""
        if self.yolo_skip_frames <= 0:
            return False
        if self._last_detection_frame_index < 0:
            return False
        return (frame_index - self._last_detection_frame_index) <= self.yolo_skip_frames

    def _detect_with_sahi(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """
        使用 SAHI 切片推理检测小目标。

        支持两种模式：
        - 逐片推理：使用 SAHI 库，简单但慢
        - 批量推理：手动切片 + YOLO 批量推理，快 4-5 倍

        Returns:
            [(x1, y1, x2, y2, confidence), ...] 原始分辨率坐标
        """
        self._init_sahi_model()
        if self._sahi_model is None:
            return []

        if self.sahi_batch_enabled:
            return self._detect_with_sahi_batch(frame)
        else:
            return self._detect_with_sahi_single(frame)

    def _detect_with_sahi_single(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """SAHI 逐片推理（使用 SAHI 库）"""
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            image=frame,
            detection_model=self._sahi_model,
            slice_height=self.sahi_slice_height,
            slice_width=self.sahi_slice_width,
            overlap_height_ratio=self.sahi_overlap,
            overlap_width_ratio=self.sahi_overlap,
            postprocess_type="NMS",
            postprocess_match_threshold=self.nms_iou,
            verbose=0,
        )

        detections = []
        for pred in result.object_prediction_list:
            if pred.category.id not in self.class_ids:
                continue
            bbox = pred.bbox
            conf = pred.score.value
            detections.append((bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, conf))

        return detections

    def _slice_image(self, frame: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
        """
        手动切片图像。

        Returns:
            (slices, coords): 切片列表 + 每个切片在原图的 (x1, y1, x2, y2) 坐标
        """
        h, w = frame.shape[:2]
        slice_w = self.sahi_slice_width
        slice_h = self.sahi_slice_height
        overlap = self.sahi_overlap

        step_w = int(slice_w * (1 - overlap))
        step_h = int(slice_h * (1 - overlap))

        slices = []
        coords = []

        y = 0
        while y < h:
            x = 0
            while x < w:
                x1 = x
                y1 = y
                x2 = min(x + slice_w, w)
                y2 = min(y + slice_h, h)

                slice_img = frame[y1:y2, x1:x2]
                slices.append(slice_img)
                coords.append((x1, y1, x2, y2))

                x += step_w
                if x >= w:
                    break
            y += step_h
            if y >= h:
                break

        return slices, coords

    def _detect_with_sahi_batch(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """
        SAHI 批量推理：手动切片 + YOLO 批量推理，比逐片快 4-5 倍。

        流程：
        1. 手动切片图像
        2. 所有切片拼成 batch 一次性送入 YOLO
        3. 坐标映射回原图
        4. NMS 去重
        """
        # Step 1: 切片
        slices, coords = self._slice_image(frame)

        if not slices:
            return []

        # Step 2: 批量推理
        batch_results = self.model(
            slices,
            device=self.device,
            classes=self.class_ids,
            conf=self.confidence,
            iou=self.nms_iou,
            max_det=50,
            verbose=False,
        )

        # Step 3: 坐标映射 + 收集所有检测框
        all_boxes = []
        all_scores = []

        for i, result in enumerate(batch_results):
            if result.boxes is None:
                continue

            x_offset, y_offset = coords[i][0], coords[i][1]

            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                # 映射回原图坐标
                x1 = float(xyxy[0]) + x_offset
                y1 = float(xyxy[1]) + y_offset
                x2 = float(xyxy[2]) + x_offset
                y2 = float(xyxy[3]) + y_offset

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(conf)

        if not all_boxes:
            return []

        # Step 4: NMS 去重
        boxes_array = np.array(all_boxes, dtype=np.float32)
        scores_array = np.array(all_scores, dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            boxes_array.tolist(),
            scores_array.tolist(),
            score_threshold=self.confidence,
            nms_threshold=self.nms_iou,
        )

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x1, y1, x2, y2 = all_boxes[idx]
                conf = all_scores[idx]
                detections.append((x1, y1, x2, y2, conf))

        return detections

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[PersonDetection]:
        """
        在单帧图像中检测人体。

        SAHI 模式：切片推理提升小目标检出率，配合 ByteTrack 跟踪。
        标准模式：直接 YOLO 推理，配合 ByteTrack 跟踪。

        Args:
            frame: BGR 格式的帧图像
            frame_index: 帧序号（用于追踪）

        Returns:
            PersonDetection 列表（含 track_id），按置信度降序排列
        """
        if self._should_skip_detection(frame_index):
            return self._cached_detections

        infer_start = time.time()
        orig_h, orig_w = frame.shape[:2]

        if self.sahi_enabled:
            # ===== SAHI 切片推理 =====
            raw_dets = self._detect_with_sahi(frame)

            # SAHI 已在原图上操作，不需要缩放
            detections: list[PersonDetection] = []
            for x1, y1, x2, y2, conf in raw_dets:
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
                if bbox.width < self.min_bbox_size or bbox.height < self.min_bbox_size:
                    continue
                detections.append(PersonDetection(
                    frame_index=frame_index,
                    timestamp=time.time(),
                    bbox=bbox,
                    track_id=None,  # SAHI 不提供跟踪，track_id 由外部管理
                ))
        else:
            # ===== 标准 YOLO 推理 =====
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
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())

                    track_id = None
                    if self.tracker_enabled and box.id is not None:
                        track_id = int(box.id[0].cpu().numpy())

                    x1 = float(xyxy[0]) * scale_x
                    y1 = float(xyxy[1]) * scale_y
                    x2 = float(xyxy[2]) * scale_x
                    y2 = float(xyxy[3]) * scale_y

                    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)

                    if bbox.width < self.min_bbox_size or bbox.height < self.min_bbox_size:
                        continue

                    detections.append(PersonDetection(
                        frame_index=frame_index,
                        timestamp=time.time(),
                        bbox=bbox,
                        track_id=track_id,
                    ))

        detections.sort(key=lambda d: d.bbox.confidence, reverse=True)

        self._cached_detections = detections
        self._last_detection_frame_index = frame_index

        infer_elapsed = time.time() - infer_start
        self._inference_times.append(infer_elapsed)
        self._total_inference_count += 1
        if len(self._inference_times) > self._inference_window_max:
            self._inference_times = self._inference_times[self._inference_window_max // 2:]

        return detections

    def detect_batch(self, frames: list[np.ndarray], start_index: int = 0) -> list[list[PersonDetection]]:
        """批量检测多帧，start_index 用于保持全局帧序号连续"""
        return [self.detect(frame, start_index + i) for i, frame in enumerate(frames)]

    @property
    def fps(self) -> float:
        """返回推理平均 FPS"""
        if not self._inference_times:
            return 0.0
        avg_time = sum(self._inference_times) / len(self._inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_fps_stats(self) -> dict:
        """返回推理详细统计"""
        if not self._inference_times:
            return {"fps": 0.0, "avg_ms": 0.0, "count": 0, "window": 0, "min_ms": 0.0, "max_ms": 0.0}
        times_ms = [t * 1000 for t in self._inference_times]
        avg_ms = sum(times_ms) / len(times_ms)
        return {
            "fps": round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0,
            "avg_ms": round(avg_ms, 2),
            "count": self._total_inference_count,
            "window": len(self._inference_times),
            "min_ms": round(min(times_ms), 2),
            "max_ms": round(max(times_ms), 2),
        }
