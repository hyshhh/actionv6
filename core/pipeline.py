"""主流水线 — 将检测、提取、分类串联"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable

import cv2
import numpy as np

from core.detector import PersonDetector
from core.frame_extractor import FrameExtractor
from core.behavior_classifier import BehaviorClassifier
from core.video_source import VideoSource, VideoSourceType
from core.camera_log import CameraBehaviorLog
from core.concurrent import ConcurrentTask, ConcurrentQueue
from utils.trackers import (
    QwenFpsTracker, FrameRateTracker, TotalFrameTracker, BitrateTracker,
)
from models.schemas import (
    FrameAnalysis,
    PersonDetection,
    BehaviorResult,
    AnalysisReport,
    Severity,
)
from utils.image_utils import draw_detections, save_image, pad_bbox, crop_region, encode_image_to_base64
from utils.logger import get_logger

logger = get_logger()


# =====================================================================
# 主流水线
# =====================================================================

class Pipeline:
    """
    行为识别主流水线。

    处理流程（每帧）：
    1. 从视频源读取帧
    2. 人体检测器识别画面中的人（支持隔帧推理）
    3. 将检测结果累积到帧缓冲区
    4. 达到指定帧数后，提取关键帧序列并调用 Qwen 分析
    5. 输出结果（显示/保存/告警回调）

    两种处理模式：
    - 级联模式（concurrent_mode=false）：每 process_every_n_frames 帧停顿等 Qwen 结果
    - 并发模式（concurrent_mode=true）：YOLO 不等 Qwen，结果异步出队按时间戳打印
    """

    def __init__(
        self,
        detector: PersonDetector,
        frame_extractor: FrameExtractor,
        classifier: BehaviorClassifier,
        video_source: VideoSource,
        process_every_n_frames: int = 5,
        buffer_size: int = 5,
        camera_interval: float = 0.1,
        alert_cooldown: int = 30,
        sustained_detection_frames: int = 1,
        max_concurrent: int = 1,
        concurrent_mode: bool = False,
        max_queued_frames: int = 50,
        output_dir: str = "output",
        save_annotated: bool = True,
        save_video: bool = False,
        save_crops: bool = True,
        save_report: bool = True,
        display: bool = True,
        display_scale: float = 0.5,
        display_input: bool = False,
        display_output: bool = True,
        camera_log_enabled: bool = True,
        camera_log_retention_hours: float = 2.0,
        camera_log_filename: str = "camera_behavior_log.json",
        alert_callback: Optional[Callable] = None,
    ):
        self.detector = detector
        self.extractor = frame_extractor
        self.classifier = classifier
        self.source = video_source
        self.process_interval = max(1, process_every_n_frames)
        self.buffer_size = max(1, buffer_size)
        self.camera_interval = camera_interval
        self.alert_cooldown = alert_cooldown
        self.sustained_detection_frames = max(1, sustained_detection_frames)
        self.max_concurrent = max(1, max_concurrent)
        self.concurrent_mode = concurrent_mode
        self.output_dir = output_dir
        self.save_video = save_video
        self.save_annotated = save_annotated and not save_video
        self.save_crops = save_crops
        self.save_report = save_report
        self.display = display
        self.display_scale = display_scale
        self.display_input = display_input
        self.display_output = display_output
        self.alert_callback = alert_callback

        self.tracking_enabled = getattr(detector, 'tracker_enabled', False)

        # ===== 内层并发线程池（Qwen API 并发，max_concurrent 控制）=====
        self._classify_executor: Optional[ThreadPoolExecutor] = None
        if self.max_concurrent >= 2:
            self._classify_executor = ThreadPoolExecutor(
                max_workers=self.max_concurrent,
                thread_name_prefix="classify",
            )

        # ===== 外层并发模式 =====
        self._concurrent_queue: Optional[ConcurrentQueue] = None
        if self.concurrent_mode:
            outer_workers = max(2, min(self.max_concurrent, 8))
            self._concurrent_queue = ConcurrentQueue(
                execute_fn=self._execute_concurrent_task,
                max_queued_frames=max_queued_frames,
                max_workers=outer_workers,
            )

        # 帧缓冲区（仅追踪模式启用）
        self._frame_buffer = None
        if self.tracking_enabled:
            from collections import deque
            self._frame_buffer = deque(maxlen=self.buffer_size)

        # 告警冷却记录
        self._alert_cooldowns: dict[str, float] = {}

        # 分析报告数据
        self._report = AnalysisReport(source="")
        self._frame_count = 0
        self._processed_count = 0
        self._consecutive_detection_count = 0

        # 摄像头行为日志
        self._camera_log: Optional[CameraBehaviorLog] = None
        is_camera_mode = self.source.source_type in (VideoSourceType.CAMERA_USB, VideoSourceType.CAMERA_RTSP)
        if camera_log_enabled and is_camera_mode:
            self._camera_log = CameraBehaviorLog(
                output_dir=output_dir,
                retention_hours=camera_log_retention_hours,
                log_filename=camera_log_filename,
            )
            logger.info(f"摄像头日志已初始化: {self._camera_log.log_path}")

        # 性能统计器
        self._qwen_fps_tracker = QwenFpsTracker()
        self._yolo_frame_rate = FrameRateTracker()
        self._bitrate_tracker = BitrateTracker()
        self._total_frame_tracker = TotalFrameTracker()

        # 视频输出
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._video_path: str = ""
        if self.save_video:
            self._video_path = os.path.join(output_dir, "output_annotated.mp4")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        if save_crops:
            os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)
        if save_annotated:
            os.makedirs(os.path.join(output_dir, "annotated"), exist_ok=True)

    # ==================================================================
    # 核心流程
    # ==================================================================

    def run(self):
        """启动主流水线"""
        mode_str = "并发模式" if self.concurrent_mode else "级联模式"
        logger.info("=" * 60)
        logger.info("行为识别流水线启动")
        logger.info(f"  处理模式: {mode_str}")
        logger.info(f"  追踪模式: {'启用' if self.tracking_enabled else '禁用'}")
        logger.info(f"  持续帧阈值: {self.sustained_detection_frames} 帧")
        logger.info(f"  处理间隔: 每 {self.process_interval} 帧")
        logger.info(f"  内层并发(Qwen): {self.max_concurrent} ({'并发' if self._classify_executor else '串行'})")
        if self.tracking_enabled:
            logger.info(f"  缓冲区大小: {self.buffer_size} 帧")
        else:
            logger.info(f"  缓冲区: 禁用（单帧分析模式）")
        if self.concurrent_mode:
            logger.info(f"  外层并发队列: 已启用")
        logger.info(f"  YOLO隔帧推理: {'每帧' if self.detector.yolo_skip_frames == 0 else f'每{self.detector.yolo_skip_frames + 1}帧1次'}")
        logger.info(f"  显示: input={self.display_input}, output={self.display_output}")
        if self._camera_log is not None:
            logger.info(f"  摄像头日志: 已启用, 保留 {self._camera_log.retention_seconds / 3600:.1f}h")
        logger.info("=" * 60)

        self._report.source = str(self.source.source_type.value)
        self._report.start_time = time.time()

        if self._concurrent_queue is not None:
            self._concurrent_queue.start_consumer()

        fps_print_interval = 2.0
        last_fps_print_time = time.time()

        try:
            last_camera_time = 0
            for frame in self.source.frames():
                current_time = time.time()
                is_camera = self.source.source_type in (
                    VideoSourceType.CAMERA_USB, VideoSourceType.CAMERA_RTSP,
                )
                if is_camera and current_time - last_camera_time < self.camera_interval:
                    continue
                last_camera_time = current_time

                self._frame_count += 1
                self._report.total_frames = self._frame_count
                self._bitrate_tracker.record(frame)

                frame_start_time = time.time()

                # Step 1: 人体检测
                detections = self.detector.detect(frame, self._frame_count)
                self._yolo_frame_rate.tick()

                # Step 2: 帧缓冲区管理
                if self._frame_buffer is not None:
                    self._frame_buffer.append((frame.copy(), detections))

                # Step 3: 持续帧计数
                if detections:
                    self._consecutive_detection_count += 1
                else:
                    self._consecutive_detection_count = 0

                # Step 4: 定期触发行为分析
                frame_analysis = None
                should_process = (
                    self._frame_count % self.process_interval == 0
                    and detections
                    and self._consecutive_detection_count >= self.sustained_detection_frames
                )

                if should_process:
                    if self.concurrent_mode:
                        self._submit_concurrent_task(frame, detections, self._frame_count)
                    else:
                        qwen_start = time.time()
                        if self.tracking_enabled:
                            frame_analysis = self._analyze_buffer(self._frame_count)
                        else:
                            frame_analysis = self._analyze_single_frame(frame, detections, self._frame_count)
                        self._qwen_fps_tracker.record(time.time() - qwen_start)
                    self._processed_count += 1

                # Step 5: 并发模式 — 尝试出队已完成的结果
                if self._concurrent_queue is not None:
                    dequeued_task = self._concurrent_queue.try_dequeue_completed()
                    if dequeued_task is not None:
                        frame_analysis = self._build_analysis_from_task(dequeued_task)

                # Step 6: 可视化
                if self.display:
                    self._render_display(frame, detections, frame_analysis)

                # 视频输出模式：每帧标注后写入视频
                if self.save_video:
                    behavior_dicts = frame_analysis.behavior_dicts if frame_analysis else None
                    annotated = draw_detections(frame, detections, behavior_dicts)
                    self._write_video_frame(annotated)

                # 记录端到端总耗时
                self._total_frame_tracker.record(time.time() - frame_start_time)

                # Step 7: 定期打印 FPS
                if current_time - last_fps_print_time >= fps_print_interval:
                    self._print_fps_stats()
                    last_fps_print_time = current_time

        except KeyboardInterrupt:
            logger.info("收到中断信号，正在退出...")

        finally:
            try:
                self._finalize()
            except Exception as e:
                logger.error(f"清理过程中发生错误: {e}")
                if self._camera_log is not None:
                    try:
                        self._camera_log.save()
                    except Exception as log_error:
                        logger.error(f"保存摄像头日志失败: {log_error}")

    # ==================================================================
    # 通用分析方法（消除重复代码的核心）
    # ==================================================================

    def _run_classification(
        self,
        analysis: FrameAnalysis,
        classify_tasks: list[tuple[int, list[str]]],
        crop_cache: dict[int, np.ndarray],
        frame: np.ndarray,
        frame_index: int,
    ) -> FrameAnalysis:
        """
        通用的分类 + 结果处理流程。

        三个分析入口（单帧、缓冲区、并发任务）共享此方法，
        仅在构建 classify_tasks 阶段有差异。
        """
        if not classify_tasks:
            return analysis

        tagged_behaviors = self._classify_concurrent(classify_tasks)
        h, w = frame.shape[:2]

        has_tracking = self.tracking_enabled and any(
            d.track_id is not None for d in analysis.detections
        )

        behaviors = []
        behavior_dicts = []

        for person_key, result in tagged_behaviors:
            person_label = f"track#{person_key}" if has_tracking else f"#{person_key}"
            logger.info(
                f"[帧 {frame_index}] 人物{person_label}: "
                f"{result.behavior_label} ({result.behavior_id}) "
                f"[{result.severity.value}]"
            )

            behaviors.append(result)
            behavior_dicts.append({
                "person_key": person_key,
                "behavior_label": result.behavior_label,
                "severity": result.severity.value,
            })

            # 保存裁剪图
            if self.save_crops:
                crop_img = crop_cache.get(person_key)
                det_idx = self._resolve_det_index(person_key, analysis.detections, has_tracking)
                if crop_img is not None and det_idx is not None and 0 <= det_idx < len(analysis.detections):
                    self._save_crop(frame, analysis.detections[det_idx], frame_index, person_key, crop_image=crop_img)

            # 告警处理
            if result.is_alert():
                self._handle_alert(result, frame_index, person_key)

            # 记录到报告
            self._record_behavior(result)

            # 摄像头日志
            if self._camera_log is not None:
                self._camera_log.add_entry(
                    frame_index=frame_index,
                    person_idx=person_key,
                    behavior_id=result.behavior_id,
                    behavior_label=result.behavior_label,
                    severity=result.severity.value,
                    description=result.description,
                )

        analysis.behaviors = behaviors
        analysis.behavior_dicts = behavior_dicts

        # 保存标注帧
        if self.save_annotated:
            annotated = draw_detections(frame, analysis.detections, behavior_dicts)
            path = os.path.join(self.output_dir, "annotated", f"frame_{frame_index:06d}.jpg")
            save_image(annotated, path)

        self._report.frame_analyses.append(analysis)
        return analysis

    @staticmethod
    def _resolve_det_index(
        person_key: int,
        detections: list[PersonDetection],
        has_tracking: bool,
    ) -> Optional[int]:
        """根据 person_key 解析对应的 detection 索引"""
        if not has_tracking:
            return person_key if isinstance(person_key, int) else None
        for i, d in enumerate(detections):
            if d.track_id == person_key:
                return i
        return None

    # ==================================================================
    # 分析入口（单帧模式 / 缓冲区模式）
    # ==================================================================

    def _analyze_single_frame(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
        frame_index: int,
    ) -> Optional[FrameAnalysis]:
        """级联模式 — 单帧分析（无追踪）"""
        if not detections:
            return None

        h, w = frame.shape[:2]
        analysis = FrameAnalysis(
            frame_index=frame_index,
            timestamp=time.time(),
            frame_width=w,
            frame_height=h,
            detections=detections,
        )

        classify_tasks, crop_cache = self._build_classify_tasks_from_detections(
            frame, detections, use_tracking=False,
        )
        return self._run_classification(analysis, classify_tasks, crop_cache, frame, frame_index)

    def _analyze_buffer(self, frame_index: int) -> Optional[FrameAnalysis]:
        """级联模式 — 缓冲区分析（追踪模式）"""
        if not self._frame_buffer:
            return None

        start_time = time.time()
        frame, last_detections = self._frame_buffer[-1]
        h, w = frame.shape[:2]

        analysis = FrameAnalysis(
            frame_index=frame_index,
            timestamp=time.time(),
            frame_width=w,
            frame_height=h,
            detections=last_detections,
        )

        # 从缓冲区提取多人关键帧
        person_keyframes = self.extractor.extract_multi_person_keyframes(
            list(self._frame_buffer),
            tracker_enabled=self.tracking_enabled,
        )

        has_tracking = self.tracking_enabled and any(
            d.track_id is not None for d in last_detections
        )

        classify_tasks: list[tuple[int, list[str]]] = []
        crop_cache: dict[int, np.ndarray] = {}

        for det_idx, det in enumerate(last_detections):
            person_key = det.track_id if has_tracking else det_idx
            keyframes_b64 = person_keyframes.get(person_key)
            if not keyframes_b64:
                continue
            classify_tasks.append((person_key, keyframes_b64))
            # 缓冲区模式下不保存裁剪图（关键帧已由 extract_multi_person_keyframes 处理）

        result = self._run_classification(analysis, classify_tasks, crop_cache, frame, frame_index)
        if result is not None:
            result.processing_time = time.time() - start_time
        return result

    # ==================================================================
    # 并发模式 — 提交与构建
    # ==================================================================

    def _submit_concurrent_task(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
        frame_index: int,
    ):
        """将当前帧构建为并发任务并入队"""
        h, w = frame.shape[:2]
        classify_tasks, crop_cache = self._build_classify_tasks_from_detections(
            frame, detections, use_tracking=self.tracking_enabled,
        )

        if self._concurrent_queue is None:
            return

        task_id = self._concurrent_queue.next_task_id()
        task = ConcurrentTask(
            task_id=task_id,
            frame_index=frame_index,
            timestamp=time.time(),
            frame=frame.copy(),
            detections=detections,
            classify_tasks=classify_tasks,
            crop_cache=crop_cache,
            tracker_enabled=self.tracking_enabled,
        )
        self._concurrent_queue.submit_task(task)

    def _execute_concurrent_task(self, task: ConcurrentTask):
        """并发任务执行回调（在线程池工作线程中运行）"""
        if task.classify_tasks:
            task.tagged_behaviors = self._classify_concurrent(task.classify_tasks)
        else:
            task.tagged_behaviors = []
        self._qwen_fps_tracker.record(task.elapsed)

    def _build_analysis_from_task(self, task: ConcurrentTask) -> FrameAnalysis:
        """从完成的并发任务构建 FrameAnalysis"""
        h, w = task.frame.shape[:2]
        analysis = FrameAnalysis(
            frame_index=task.frame_index,
            timestamp=task.timestamp,
            frame_width=w,
            frame_height=h,
            detections=task.detections,
        )

        # 用通用方法处理结果
        has_tracking = task.tracker_enabled and any(
            d.track_id is not None for d in task.detections
        )

        behaviors = []
        behavior_dicts = []

        for person_key, result in task.tagged_behaviors:
            person_label = f"track#{person_key}" if has_tracking else f"#{person_key}"
            logger.info(
                f"[帧 {task.frame_index}] 人物{person_label}: "
                f"{result.behavior_label} ({result.behavior_id}) "
                f"[{result.severity.value}]"
            )

            behaviors.append(result)
            behavior_dicts.append({
                "person_key": person_key,
                "behavior_label": result.behavior_label,
                "severity": result.severity.value,
            })

            # 保存裁剪图
            if self.save_crops:
                crop_img = task.crop_cache.get(person_key)
                det_idx = self._resolve_det_index(person_key, task.detections, has_tracking)
                if crop_img is not None and det_idx is not None and 0 <= det_idx < len(task.detections):
                    self._save_crop(task.frame, task.detections[det_idx], task.frame_index, person_key, crop_image=crop_img)

            if result.is_alert():
                self._handle_alert(result, task.frame_index, person_key)

            self._record_behavior(result)

            if self._camera_log is not None:
                self._camera_log.add_entry(
                    frame_index=task.frame_index,
                    person_idx=person_key,
                    behavior_id=result.behavior_id,
                    behavior_label=result.behavior_label,
                    severity=result.severity.value,
                    description=result.description,
                )

        analysis.behaviors = behaviors
        analysis.behavior_dicts = behavior_dicts
        analysis.processing_time = task.elapsed

        if self.save_annotated:
            annotated = draw_detections(task.frame, task.detections, behavior_dicts)
            path = os.path.join(self.output_dir, "annotated", f"frame_{task.frame_index:06d}.jpg")
            save_image(annotated, path)

        self._report.frame_analyses.append(analysis)
        return analysis

    # ==================================================================
    # 视频输出
    # ==================================================================

    def _ensure_video_writer(self, frame: np.ndarray):
        """延迟初始化 VideoWriter（首帧时根据分辨率和源帧率创建）"""
        if self._video_writer is not None:
            return

        h, w = frame.shape[:2]
        # 从视频源获取帧率，摄像头默认 25fps
        fps = 25.0
        if self.source._cap is not None:
            src_fps = self.source._cap.get(cv2.CAP_PROP_FPS)
            if src_fps > 0:
                fps = src_fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(self._video_path, fourcc, fps, (w, h))
        logger.info(f"视频输出已初始化: {self._video_path} ({w}x{h} @ {fps:.1f}fps)")

    def _write_video_frame(self, frame: np.ndarray):
        """将一帧写入输出视频"""
        self._ensure_video_writer(frame)
        if self._video_writer is not None:
            self._video_writer.write(frame)

    def _release_video_writer(self):
        """释放 VideoWriter"""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            logger.info(f"输出视频已保存: {self._video_path}")

    # ==================================================================
    # 辅助方法
    # ==================================================================

    def _build_classify_tasks_from_detections(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
        use_tracking: bool = False,
    ) -> tuple[list[tuple[int, list[str]]], dict[int, np.ndarray]]:
        """从检测结果构建分类任务和裁剪缓存（三段重复代码的统一提取点）"""
        h, w = frame.shape[:2]
        classify_tasks: list[tuple[int, list[str]]] = []
        crop_cache: dict[int, np.ndarray] = {}

        has_tracking = use_tracking and any(
            d.track_id is not None for d in detections
        )

        for det_idx, det in enumerate(detections):
            person_key = det.track_id if has_tracking and det.track_id is not None else det_idx

            bw = det.bbox.x2 - det.bbox.x1
            bh = det.bbox.y2 - det.bbox.y1
            pad_ratio = self.extractor._get_padding(bw, bh, w, h)
            px1, py1, px2, py2 = pad_bbox(
                det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2,
                pad_ratio, w, h,
            )
            crop = crop_region(frame, px1, py1, px2, py2)
            if crop is None:
                continue

            crop_b64 = encode_image_to_base64(crop, fmt=".jpg", quality=80)
            classify_tasks.append((person_key, [crop_b64]))
            crop_cache[person_key] = crop

        return classify_tasks, crop_cache

    def _classify_concurrent(
        self,
        tasks: list[tuple[int, list[str]]],
    ) -> list[tuple[int, BehaviorResult]]:
        """
        内层并发调用行为分类器，结果严格按输入顺序返回。
        """
        if not tasks:
            return []

        if self._classify_executor is None:
            results = []
            for person_key, keyframes_b64 in tasks:
                result = self.classifier.classify(keyframes_b64)
                results.append((person_key, result))
            return results

        futures_map = {}
        for idx, (person_key, keyframes_b64) in enumerate(tasks):
            future = self._classify_executor.submit(self.classifier.classify, keyframes_b64)
            futures_map[future] = (idx, person_key)

        indexed_results: dict[int, tuple[int, BehaviorResult]] = {}
        for future in as_completed(futures_map):
            idx, person_key = futures_map[future]
            try:
                result = future.result()
                indexed_results[idx] = (person_key, result)
            except Exception as e:
                logger.error(f"并发分类失败 (person_key={person_key}): {e}")
                indexed_results[idx] = (person_key, BehaviorResult(
                    behavior_id="unknown",
                    behavior_label="未知",
                    description=f"并发分类异常: {str(e)}",
                    severity=Severity.NORMAL,
                ))

        return [indexed_results[i] for i in sorted(indexed_results.keys())]

    def _save_crop(
        self,
        frame: np.ndarray,
        detection: PersonDetection,
        frame_index: int,
        person_idx: int,
        crop_image: Optional[np.ndarray] = None,
    ):
        """保存人体裁剪图"""
        if crop_image is not None:
            crop = crop_image
        else:
            h, w = frame.shape[:2]
            bbox = detection.bbox
            bw = bbox.x2 - bbox.x1
            bh = bbox.y2 - bbox.y1
            pad_ratio = self.extractor._get_padding(bw, bh, w, h)
            px1, py1, px2, py2 = pad_bbox(
                bbox.x1, bbox.y1, bbox.x2, bbox.y2,
                pad_ratio, w, h,
            )
            crop = crop_region(frame, px1, py1, px2, py2)

        if crop is not None:
            path = os.path.join(
                self.output_dir, "crops",
                f"frame{frame_index:06d}_person{person_idx}.jpg"
            )
            save_image(crop, path)

    def _handle_alert(
        self,
        result: BehaviorResult,
        frame_index: int,
        person_idx: int,
    ):
        """处理告警（含冷却机制）"""
        now = time.time()
        last_alert = self._alert_cooldowns.get(result.behavior_id, 0)

        if now - last_alert < self.alert_cooldown:
            logger.debug(
                f"告警冷却中: {result.behavior_id}, "
                f"剩余 {self.alert_cooldown - (now - last_alert):.0f}s"
            )
            return

        self._alert_cooldowns[result.behavior_id] = now

        alert_msg = (
            f"告警! [帧 {frame_index}] 人物#{person_idx}: "
            f"{result.behavior_label} ({result.severity.value}) - {result.description}"
        )
        logger.warning(alert_msg)

        self._report.alerts.append({
            "frame_index": frame_index,
            "person_idx": person_idx,
            "behavior_id": result.behavior_id,
            "behavior_label": result.behavior_label,
            "severity": result.severity.value,
            "description": result.description,
            "timestamp": time.time(),
        })

        if self.alert_callback:
            try:
                self.alert_callback(result, frame_index, person_idx)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")

    def _record_behavior(self, result: BehaviorResult):
        """记录行为到报告统计"""
        bid = result.behavior_id
        self._report.behavior_counts[bid] = self._report.behavior_counts.get(bid, 0) + 1

    # ==================================================================
    # 可视化与显示
    # ==================================================================

    def _render_display(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
        analysis: Optional[FrameAnalysis] = None,
    ):
        """渲染显示画面"""
        show_input = self.display_input
        show_output = self.display_output

        if not show_input and not show_output:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
            return

        if show_output:
            annotated = self._draw_frame(frame, detections, analysis)
        else:
            annotated = None

        if show_input:
            input_frame = frame.copy()
            info = f"Input | Frame: {self._frame_count}"
            cv2.putText(input_frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(input_frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if show_input and show_output and annotated is not None:
            ih, iw = input_frame.shape[:2]
            oh, ow = annotated.shape[:2]
            target_h = min(ih, oh)
            if ih != target_h:
                input_frame = cv2.resize(input_frame, (int(iw * target_h / ih), target_h))
            if oh != target_h:
                annotated = cv2.resize(annotated, (int(ow * target_h / oh), target_h))
            display_frame = np.hstack([input_frame, annotated])
            window_title = "Behavior Recognition Agent [Input | Output]"
        elif show_output and annotated is not None:
            display_frame = annotated
            window_title = "Behavior Recognition Agent [Output]"
        elif show_input:
            display_frame = input_frame
            window_title = "Behavior Recognition Agent [Input]"
        else:
            return

        if self.display_scale != 1.0:
            h, w = display_frame.shape[:2]
            new_w = int(w * self.display_scale)
            new_h = int(h * self.display_scale)
            display_frame = cv2.resize(display_frame, (new_w, new_h))

        cv2.imshow(window_title, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logger.info("用户按 'q' 退出")
            raise KeyboardInterrupt
        elif key == ord("s"):
            path = os.path.join(self.output_dir, f"screenshot_{self._frame_count}.jpg")
            save_image(display_frame, path)
            logger.info(f"截图已保存: {path}")

    def _draw_frame(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
        analysis: Optional[FrameAnalysis] = None,
    ) -> np.ndarray:
        """绘制带检测框和行为标签的帧"""
        behavior_dicts = None
        if analysis and analysis.behavior_dicts:
            behavior_dicts = analysis.behavior_dicts

        annotated = draw_detections(frame, detections, behavior_dicts)

        info = f"Frame: {self._frame_count} | Persons: {len(detections)} | Processed: {self._processed_count}"
        cv2.putText(annotated, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.putText(
            annotated,
            f"Sustained: {self._consecutive_detection_count}/{self.sustained_detection_frames}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1,
        )

        track_label = "Track: ON" if self.tracking_enabled else "Track: OFF"
        cv2.putText(
            annotated, track_label, (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 200, 200) if self.tracking_enabled else (128, 128, 128), 1,
        )

        yolo_stats = self.detector.get_fps_stats()
        yolo_fr = self._yolo_frame_rate.get_stats()
        qwen_stats = self._qwen_fps_tracker.get_stats()
        stream_stats = self._bitrate_tracker.get_stats()
        total_stats = self._total_frame_tracker.get_stats()
        fps_text = (
            f"Loop: {yolo_fr['frame_rate']:.1f} frames/s | "
            f"YOLO infer: {yolo_stats['avg_ms']:.1f}ms | "
            f"Stream: {stream_stats['mbps']:.1f} MB/s | "
            f"Qwen: {qwen_stats['throughput']:.2f} req/s "
            f"(avg {qwen_stats['avg_ms']:.0f}ms) | "
            f"Total: {total_stats['avg_ms']:.0f}ms/frame ({total_stats['fps']:.1f} fps)"
        )
        cv2.putText(annotated, fps_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        mode_text = f"Mode: {'Concurrent' if self.concurrent_mode else 'Cascade'} | YOLO skip: {self.detector.yolo_skip_frames}"
        cv2.putText(annotated, mode_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        if analysis:
            for b in analysis.behaviors:
                if b.is_alert():
                    alert_text = f"ALERT: {b.behavior_label}!"
                    cv2.putText(
                        annotated, alert_text, (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
                    )

        return annotated

    # ==================================================================
    # FPS 打印
    # ==================================================================

    def _print_fps_stats(self):
        """在终端和日志中打印性能统计"""
        yolo_stats = self.detector.get_fps_stats()
        yolo_fr = self._yolo_frame_rate.get_stats()
        qwen_stats = self._qwen_fps_tracker.get_stats()
        stream_stats = self._bitrate_tracker.get_stats()
        total_stats = self._total_frame_tracker.get_stats()

        fps_line = (
            f"[Speed] Loop: {yolo_fr['frame_rate']:6.1f} frames/s "
            f"| YOLO infer: {yolo_stats['avg_ms']:6.1f}ms "
            f"(min={yolo_stats['min_ms']:.1f} max={yolo_stats['max_ms']:.1f}ms) "
            f"| Stream: {stream_stats['mbps']:7.2f} MB/s "
            f"| Qwen: {qwen_stats['throughput']:6.2f} req/s "
            f"(avg {qwen_stats['avg_ms']:6.1f}ms, count={qwen_stats['count']}) "
            f"| Total/frame: {total_stats['avg_ms']:6.1f}ms "
            f"(min={total_stats['min_ms']:.1f} max={total_stats['max_ms']:.1f}ms, "
            f"{total_stats['fps']:.1f} frames/s)"
        )
        logger.info(fps_line)

    # ==================================================================
    # 收尾
    # ==================================================================

    def _finalize(self):
        """流水线结束，保存报告"""
        self._report.end_time = time.time()
        self._report.processed_frames = self._processed_count

        # 关闭外层并发
        if self._concurrent_queue is not None:
            self._concurrent_queue.shutdown()

        # 关闭内层并发线程池
        if self._classify_executor is not None:
            try:
                self._classify_executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                self._classify_executor.shutdown(wait=True)
            logger.info("内层并发线程池已关闭")

        # 保存摄像头行为日志
        if self._camera_log is not None:
            self._camera_log.save()
            logger.info(
                f"摄像头行为日志已保存: {self._camera_log.log_path} "
                f"({self._camera_log.entry_count} 条)"
            )

        # 释放视频源
        try:
            self.source.release()
        except Exception as e:
            logger.error(f"释放视频源失败: {e}")

        if self.display:
            cv2.destroyAllWindows()

        # 释放视频输出
        self._release_video_writer()

        # 保存报告
        if self.save_report:
            report_path = os.path.join(self.output_dir, "analysis_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self._report.summary(), f, ensure_ascii=False, indent=2)
            logger.info(f"分析报告已保存: {report_path}")

            if self._report.alerts:
                alerts_path = os.path.join(self.output_dir, "alerts.json")
                with open(alerts_path, "w", encoding="utf-8") as f:
                    json.dump(self._report.alerts, f, ensure_ascii=False, indent=2)
                logger.info(f"告警记录已保存: {alerts_path} ({len(self._report.alerts)} 条)")

        # 打印最终摘要
        summary = self._report.summary()
        yolo_stats = self.detector.get_fps_stats()
        yolo_fr = self._yolo_frame_rate.get_stats()
        qwen_stats = self._qwen_fps_tracker.get_stats()
        stream_stats = self._bitrate_tracker.get_stats()
        total_stats = self._total_frame_tracker.get_stats()

        logger.info("=" * 60)
        logger.info("分析完成! 摘要:")
        logger.info(f"  输入源: {summary['source']}")
        logger.info(f"  运行时长: {summary['duration_seconds']}s")
        logger.info(f"  总帧数: {summary['total_frames']}")
        logger.info(f"  分析帧数: {summary['processed_frames']}")
        logger.info(f"  总检测数: {summary['total_detections']}")
        logger.info(f"  行为统计: {summary['behavior_counts']}")
        logger.info(f"  告警次数: {summary['alert_count']}")
        logger.info(f"  处理模式: {'并发' if self.concurrent_mode else '级联'}")
        logger.info(f"  Loop: {yolo_fr['frame_rate']:.1f} frames/s | YOLO infer: avg={yolo_stats['avg_ms']:.1f}ms min={yolo_stats['min_ms']:.1f} max={yolo_stats['max_ms']:.1f}ms ({yolo_stats['count']} 次)")
        logger.info(f"  Stream: {stream_stats['mbps']:.2f} MB/s (total {stream_stats['total_mb']:.1f} MB, avg {stream_stats.get('avg_frame_kb', 0):.1f} KB/frame)")
        logger.info(f"  Qwen: {qwen_stats['throughput']:.2f} req/s (avg {qwen_stats['avg_ms']:.1f}ms, {qwen_stats['count']} 次)")
        logger.info(f"  Total/frame: avg={total_stats['avg_ms']:.1f}ms min={total_stats['min_ms']:.1f}ms max={total_stats['max_ms']:.1f}ms ({total_stats['fps']:.1f} frames/s, {total_stats['count']} 帧)")
        if self._camera_log is not None:
            logger.info(f"  日志条目: {self._camera_log.entry_count}")
        logger.info("=" * 60)
