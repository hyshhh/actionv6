"""主流水线 — 将检测、提取、分类串联"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from typing import Optional, Callable

import cv2
import numpy as np

from core.detector import PersonDetector
from core.frame_extractor import FrameExtractor
from core.behavior_classifier import BehaviorClassifier
from core.video_source import VideoSource, VideoSourceType
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
# 并发模式任务封装
# =====================================================================

class _ConcurrentTask:
    """
    外层并发任务封装（功能2）。

    每个 task 代表一次 process_every_n_frames 触发点的完整处理：
    - 包含该帧的图像、检测结果、帧序号
    - 构建好的 classify_tasks（裁剪后的 base64 + person_key）
    - 由外层线程池异步调度执行 Qwen 推理
    - 结果回填后按 frame_index 顺序出队
    """

    def __init__(
        self,
        task_id: int,
        frame_index: int,
        timestamp: float,
        frame: np.ndarray,
        detections: list[PersonDetection],
        classify_tasks: list[tuple[int, list[str]]],
        crop_cache: dict[int, np.ndarray],
        tracker_enabled: bool,
    ):
        self.task_id = task_id
        self.frame_index = frame_index
        self.timestamp = timestamp
        self.frame = frame
        self.detections = detections
        self.classify_tasks = classify_tasks
        self.crop_cache = crop_cache
        self.tracker_enabled = tracker_enabled

        # 推理完成后回填
        self.tagged_behaviors: list[tuple[int, BehaviorResult]] = []
        self.completed = False
        self.error: Optional[Exception] = None


# =====================================================================
# 摄像头行为日志
# =====================================================================

class CameraBehaviorLog:
    """
    摄像头行为日志管理器。

    功能：
    - 记录每次行为识别结果，包含时间戳和行为信息
    - 定期清理超过保留时长的日志条目
    - 以 JSON 格式持久化存储
    """

    def __init__(
        self,
        output_dir: str,
        retention_hours: float = 2.0,
        log_filename: str = "camera_behavior_log.json",
    ):
        self.output_dir = output_dir
        self.retention_seconds = retention_hours * 3600
        self.log_path = os.path.join(output_dir, log_filename)

        self._entries: list[dict] = []
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
        result: BehaviorResult,
    ):
        now = time.time()
        entry = {
            "timestamp": round(now, 3),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "frame_index": frame_index,
            "person_idx": person_idx,
            "behavior_id": result.behavior_id,
            "behavior_label": result.behavior_label,
            "severity": result.severity.value,
            "description": result.description,
        }
        self._entries.append(entry)
        self.save()

    def _cleanup(self):
        now = time.time()
        cutoff = now - self.retention_seconds
        before_count = len(self._entries)
        self._entries = [e for e in self._entries if e.get("timestamp", 0) >= cutoff]
        removed = before_count - len(self._entries)
        if removed > 0:
            logger.debug(f"摄像头日志清理: 删除 {removed} 条过期记录")

    def save(self):
        self._cleanup()
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
            logger.info(f"摄像头日志文件已写入: {self.log_path}")
        except Exception as e:
            logger.error(f"保存摄像头日志失败: {e}")

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# =====================================================================
# Qwen FPS 统计器
# =====================================================================

class QwenFpsTracker:
    """
    功能3：Qwen 推理速度统计。

    Qwen 推理频率与 YOLO 不同：
    - YOLO 根据每帧（或隔帧）的推理时间算 FPS
    - Qwen 根据每次 process_every_n_frames 触发时的推理时间算 FPS
    """

    def __init__(self):
        self._inference_times: list[float] = []
        self._total_count: int = 0

    def record(self, elapsed: float):
        self._inference_times.append(elapsed)
        self._total_count += 1

    @property
    def fps(self) -> float:
        if not self._inference_times:
            return 0.0
        # 使用最近 N 次的滑动窗口均值（最近50次，防止统计漂移）
        window = self._inference_times[-50:]
        avg_time = sum(window) / len(window)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_stats(self) -> dict:
        if not self._inference_times:
            return {"fps": 0.0, "avg_ms": 0.0, "count": 0, "min_ms": 0.0, "max_ms": 0.0}
        window = self._inference_times[-50:]
        times_ms = [t * 1000 for t in window]
        avg_ms = sum(times_ms) / len(times_ms)
        return {
            "fps": round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0,
            "avg_ms": round(avg_ms, 2),
            "count": self._total_count,
            "min_ms": round(min(times_ms), 2),
            "max_ms": round(max(times_ms), 2),
        }


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
    - 并发模式（concurrent_mode=true）：YOLO 不等 Qwen，结果异步出队按时间戳打印（功能2）
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
        """
        Args:
            detector: 人体检测器（内置追踪器开关、隔帧推理）
            frame_extractor: 关键帧提取器
            classifier: 行为分类器
            video_source: 视频输入源
            process_every_n_frames: 每 N 帧触发一次行为分析
            buffer_size: 历史帧缓冲区大小（仅追踪模式生效）
            camera_interval: 摄像头调用间隔（秒）
            alert_cooldown: 同一行为告警冷却秒数
            sustained_detection_frames: 连续N帧检测到目标才触发API
            max_concurrent: 最大并发 API 请求数（内层并发，1=串行）
            concurrent_mode: 是否启用外层并发模式（功能2）
            max_queued_frames: 并发模式下最大队列帧数
            output_dir: 输出目录
            save_annotated: 是否保存标注帧
            save_crops: 是否保存人体裁剪图
            save_report: 是否保存分析报告
            display: 是否显示实时画面
            display_scale: 视频窗口缩放比例
            display_input: 是否显示输入源画面（功能4）
            display_output: 是否显示输出画面（功能4）
            camera_log_enabled: 是否启用摄像头行为日志
            camera_log_retention_hours: 日志保留时长
            camera_log_filename: 日志文件名
            alert_callback: 告警回调函数
        """
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
        self.max_queued_frames = max_queued_frames
        self.output_dir = output_dir
        self.save_annotated = save_annotated
        self.save_crops = save_crops
        self.save_report = save_report
        self.display = display
        self.display_scale = display_scale
        self.display_input = display_input
        self.display_output = display_output
        self.alert_callback = alert_callback

        # 追踪模式判断
        self.tracking_enabled = getattr(detector, 'tracker_enabled', False)

        # ===== 内层并发线程池（Qwen API 并发，max_concurrent 控制）=====
        self._classify_executor: Optional[ThreadPoolExecutor] = None
        if self.max_concurrent >= 2:
            self._classify_executor = ThreadPoolExecutor(
                max_workers=self.max_concurrent,
                thread_name_prefix="classify",
            )

        # ===== 功能2：外层并发模式相关 =====
        self._concurrent_executor: Optional[ThreadPoolExecutor] = None
        self._pending_task_queue: Queue[_ConcurrentTask] = Queue(maxsize=max_queued_frames)
        self._completed_results: dict[int, _ConcurrentTask] = {}  # task_id → completed task
        self._next_display_task_id: int = 0  # 下一个要显示的任务 ID
        self._task_id_counter: int = 0  # 任务 ID 自增计数器
        self._concurrent_lock = threading.Lock()

        if self.concurrent_mode:
            # 外层并发：每个 process_every_n_frames 到来时，提交一个 worker 处理
            # 使用独立线程池，与内层 max_concurrent（API并发）互不干扰
            outer_workers = max(2, min(self.max_concurrent, 8))
            self._concurrent_executor = ThreadPoolExecutor(
                max_workers=outer_workers,
                thread_name_prefix="concurrent-qwen",
            )
            # 启动外层消费线程
            self._consumer_stop_event = threading.Event()
            self._consumer_thread = threading.Thread(
                target=self._concurrent_consumer_loop,
                daemon=True,
                name="concurrent-consumer",
            )

        # 帧缓冲区（仅追踪模式启用）
        self._frame_buffer: Optional[deque[tuple[np.ndarray, list[PersonDetection]]]] = None
        if self.tracking_enabled:
            self._frame_buffer = deque(maxlen=self.buffer_size)

        # 告警冷却记录
        self._alert_cooldowns: dict[str, float] = {}

        # 分析报告数据
        self._report = AnalysisReport(source="")
        self._frame_count = 0
        self._processed_count = 0

        # 持续帧计数器
        self._consecutive_detection_count = 0

        # 摄像头行为日志
        self._camera_log: Optional[CameraBehaviorLog] = None
        self._camera_log_enabled = camera_log_enabled
        self._camera_log_retention_hours = camera_log_retention_hours
        self._camera_log_filename = camera_log_filename

        # 功能3：Qwen FPS 统计
        self._qwen_fps_tracker = QwenFpsTracker()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        if save_crops:
            os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)
        if save_annotated:
            os.makedirs(os.path.join(output_dir, "annotated"), exist_ok=True)

        # 摄像头模式日志初始化
        is_camera_mode = self.source.source_type in (VideoSourceType.CAMERA_USB, VideoSourceType.CAMERA_RTSP)
        if self._camera_log_enabled and is_camera_mode:
            self._camera_log = CameraBehaviorLog(
                output_dir=output_dir,
                retention_hours=camera_log_retention_hours,
                log_filename=camera_log_filename,
            )
            logger.info(f"摄像头日志已初始化: {self._camera_log.log_path}")

    # ==================================================================
    # 功能2：外层并发 — 消费者循环
    # ==================================================================

    def _concurrent_consumer_loop(self):
        """
        外层并发消费者线程。

        持续从 _pending_task_queue 取出任务，提交到外层线程池执行 Qwen 推理。
        推理完成后将结果存入 _completed_results，等待按帧序号顺序出队显示。
        """
        while not self._consumer_stop_event.is_set():
            try:
                task = self._pending_task_queue.get(timeout=0.5)
            except Empty:
                continue

            # 提交到外层线程池执行
            if self._concurrent_executor is not None:
                self._concurrent_executor.submit(self._execute_concurrent_task, task)
            else:
                self._execute_concurrent_task(task)

    def _execute_concurrent_task(self, task: _ConcurrentTask):
        """
        执行单个并发任务（在工作线程中运行）。

        内层并发已经由 _classify_concurrent 实现（max_concurrent 控制），
        这里负责：
        1. 调用内层并发分类
        2. 记录 Qwen 推理耗时（功能3）
        3. 将结果存入 _completed_results
        """
        try:
            qwen_start = time.time()

            if task.classify_tasks:
                task.tagged_behaviors = self._classify_concurrent(task.classify_tasks)
            else:
                task.tagged_behaviors = []

            qwen_elapsed = time.time() - qwen_start
            self._qwen_fps_tracker.record(qwen_elapsed)

            task.completed = True

        except Exception as e:
            logger.error(f"并发任务执行失败 (task_id={task.task_id}): {e}")
            task.error = e
            task.completed = True

        # 存入完成结果字典
        with self._concurrent_lock:
            self._completed_results[task.task_id] = task

    def _try_dequeue_completed_task(self) -> Optional[_ConcurrentTask]:
        """
        尝试按帧序号顺序出队已完成的任务（功能2）。

        严格保证按时间戳 / frame_index 顺序输出：
        只有 _next_display_task_id 对应的任务完成时才出队。
        """
        with self._concurrent_lock:
            task = self._completed_results.get(self._next_display_task_id)
            if task is not None and task.completed:
                del self._completed_results[self._next_display_task_id]
                self._next_display_task_id += 1
                return task
        return None

    # ==================================================================
    # 核心流程
    # ==================================================================

    def run(self):
        """
        启动主流水线。

        支持两种处理模式：
        - 级联模式（concurrent_mode=False）：按 process_every_n_frames 间隔同步调用 Qwen
        - 并发模式（concurrent_mode=True）：YOLO 不等 Qwen，结果异步出队按时间戳打印
        """
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
            logger.info(f"  外层并发队列: 最大 {self.max_queued_frames} 帧")
        logger.info(f"  YOLO隔帧推理: {'每帧' if self.detector.yolo_skip_frames == 0 else f'每{self.detector.yolo_skip_frames + 1}帧1次'}")
        logger.info(f"  显示: input={self.display_input}, output={self.display_output}")
        if self._camera_log is not None:
            logger.info(f"  摄像头日志: 已启用, 保留 {self._camera_log_retention_hours}h")
        logger.info("=" * 60)

        self._report.source = str(self.source.source_type.value)
        self._report.start_time = time.time()

        # 启动外层并发消费者线程
        if self.concurrent_mode and hasattr(self, '_consumer_thread'):
            self._consumer_thread.start()

        fps_print_interval = 2.0  # 每2秒打印一次 FPS
        last_fps_print_time = time.time()

        try:
            last_camera_time = 0
            for frame in self.source.frames():
                # 控制摄像头调用间隔
                current_time = time.time()
                is_camera = self.source.source_type in (
                    VideoSourceType.CAMERA_USB, VideoSourceType.CAMERA_RTSP,
                )
                if is_camera:
                    if current_time - last_camera_time < self.camera_interval:
                        continue
                last_camera_time = current_time

                self._frame_count += 1
                self._report.total_frames = self._frame_count

                # Step 1: 人体检测（含隔帧推理）
                detections = self.detector.detect(frame, self._frame_count)

                # Step 2: 帧缓冲区管理
                if self.tracking_enabled and self._frame_buffer is not None:
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
                        # 功能2：并发模式 — 提交任务到队列，不等待结果
                        self._submit_concurrent_task(frame, detections, self._frame_count)
                    else:
                        # 级联模式 — 同步等待 Qwen 结果
                        qwen_start = time.time()
                        if self.tracking_enabled:
                            frame_analysis = self._analyze_buffer(self._frame_count)
                        else:
                            frame_analysis = self._analyze_single_frame(frame, detections, self._frame_count)
                        qwen_elapsed = time.time() - qwen_start
                        self._qwen_fps_tracker.record(qwen_elapsed)
                    self._processed_count += 1

                # Step 5: 并发模式 — 尝试出队已完成的结果
                if self.concurrent_mode:
                    dequeued_task = self._try_dequeue_completed_task()
                    if dequeued_task is not None:
                        frame_analysis = self._build_analysis_from_task(dequeued_task)

                # Step 6: 可视化
                if self.display:
                    self._render_display(frame, detections, frame_analysis)

                # Step 7: 功能3 — 定期打印 FPS
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
    # 功能2：并发模式 — 提交任务
    # ==================================================================

    def _submit_concurrent_task(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
        frame_index: int,
    ):
        """
        功能2：将当前帧的检测结果构建为并发任务并入队。

        步骤：
        1. 为每个人裁剪人体区域并编码为 base64
        2. 构建 _ConcurrentTask
        3. 入队 _pending_task_queue（消费者线程异步处理 Qwen 推理）
        """
        h, w = frame.shape[:2]

        classify_tasks: list[tuple[int, list[str]]] = []
        crop_cache: dict[int, np.ndarray] = {}

        has_tracking = self.tracking_enabled and any(
            d.track_id is not None for d in detections
        )

        for det_idx, det in enumerate(detections):
            if has_tracking and det.track_id is not None:
                person_key = det.track_id
            else:
                person_key = det_idx

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

        with self._concurrent_lock:
            task_id = self._task_id_counter
            self._task_id_counter += 1

        task = _ConcurrentTask(
            task_id=task_id,
            frame_index=frame_index,
            timestamp=time.time(),
            frame=frame.copy(),
            detections=detections,
            classify_tasks=classify_tasks,
            crop_cache=crop_cache,
            tracker_enabled=has_tracking,
        )

        try:
            self._pending_task_queue.put_nowait(task)
            logger.debug(f"并发任务入队: task_id={task_id}, frame={frame_index}, crops={len(classify_tasks)}")
        except Exception:
            logger.warning(f"并发队列已满，丢弃帧 {frame_index} 的任务")

    def _build_analysis_from_task(self, task: _ConcurrentTask) -> FrameAnalysis:
        """从完成的并发任务构建 FrameAnalysis"""
        h, w = task.frame.shape[:2]
        analysis = FrameAnalysis(
            frame_index=task.frame_index,
            timestamp=task.timestamp,
            frame_width=w,
            frame_height=h,
            detections=task.detections,
        )

        behaviors = []
        behavior_dicts = []

        has_tracking = task.tracker_enabled and any(
            d.track_id is not None for d in task.detections
        )

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
            det_idx = person_key if not has_tracking else next(
                (i for i, d in enumerate(task.detections) if d.track_id == person_key), person_key
            )
            if self.save_crops and isinstance(det_idx, int) and 0 <= det_idx < len(task.detections):
                crop_img = task.crop_cache.get(person_key)
                self._save_crop(task.frame, task.detections[det_idx], task.frame_index, person_key, crop_image=crop_img)

            # 告警处理
            if result.is_alert():
                self._handle_alert(result, task.frame_index, person_key)

            # 记录到报告
            self._record_behavior(result, task.frame_index)

            # 摄像头日志
            if self._camera_log is not None:
                self._camera_log.add_entry(
                    frame_index=task.frame_index,
                    person_idx=person_key,
                    result=result,
                )

        analysis.behaviors = behaviors
        analysis.behavior_dicts = behavior_dicts

        # 保存标注帧
        if self.save_annotated:
            annotated = draw_detections(task.frame, task.detections, behavior_dicts)
            path = os.path.join(self.output_dir, "annotated", f"frame_{task.frame_index:06d}.jpg")
            save_image(annotated, path)

        self._report.frame_analyses.append(analysis)
        return analysis

    # ==================================================================
    # 级联模式分析方法（原有逻辑）
    # ==================================================================

    def _classify_concurrent(
        self,
        tasks: list[tuple[int, list[str]]],
    ) -> list[tuple[int, BehaviorResult]]:
        """
        内层并发调用行为分类器，结果严格按输入顺序返回。

        当 max_concurrent >= 2 时，使用 ThreadPoolExecutor 并发提交 API 请求；
        当 max_concurrent == 1 时，退化为串行模式。
        """
        if not tasks:
            return []

        # 串行模式
        if self._classify_executor is None:
            results = []
            for person_key, keyframes_b64 in tasks:
                result = self.classifier.classify(keyframes_b64)
                results.append((person_key, result))
            return results

        # 并发模式
        semaphore = threading.Semaphore(self.max_concurrent)
        futures_map = {}

        def _call_with_semaphore(keyframes: list[str]) -> BehaviorResult:
            semaphore.acquire()
            try:
                return self.classifier.classify(keyframes)
            finally:
                semaphore.release()

        for idx, (person_key, keyframes_b64) in enumerate(tasks):
            future = self._classify_executor.submit(_call_with_semaphore, keyframes_b64)
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

        ordered_results = [
            indexed_results[i]
            for i in sorted(indexed_results.keys())
        ]

        return ordered_results

    def _analyze_single_frame(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
        frame_index: int,
    ) -> Optional[FrameAnalysis]:
        """
        级联模式 — 单帧分析（无追踪）。
        """
        if not detections:
            return None

        start_time = time.time()
        h, w = frame.shape[:2]

        analysis = FrameAnalysis(
            frame_index=frame_index,
            timestamp=time.time(),
            frame_width=w,
            frame_height=h,
            detections=detections,
        )

        classify_tasks: list[tuple[int, list[str]]] = []
        crop_cache: dict[int, np.ndarray] = {}

        for det_idx, det in enumerate(detections):
            person_key = det_idx

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

        if not classify_tasks:
            return analysis

        tagged_behaviors = self._classify_concurrent(classify_tasks)

        behaviors = []
        behavior_dicts = []

        for person_key, result in tagged_behaviors:
            logger.info(
                f"[帧 {frame_index}] 人物#{person_key}: "
                f"{result.behavior_label} ({result.behavior_id}) "
                f"[{result.severity.value}]"
            )

            det_idx = person_key
            if det_idx < len(detections) and self.save_crops:
                crop_img = crop_cache.get(person_key)
                if crop_img is not None:
                    self._save_crop(frame, detections[det_idx], frame_index, person_key, crop_image=crop_img)

            if result.is_alert():
                self._handle_alert(result, frame_index, person_key)

            self._record_behavior(result, frame_index)

            if self._camera_log is not None:
                self._camera_log.add_entry(
                    frame_index=frame_index,
                    person_idx=person_key,
                    result=result,
                )

            behaviors.append(result)
            behavior_dicts.append({
                "person_key": person_key,
                "behavior_label": result.behavior_label,
                "severity": result.severity.value,
            })

        analysis.behaviors = behaviors
        analysis.behavior_dicts = behavior_dicts
        analysis.processing_time = time.time() - start_time

        if self.save_annotated:
            annotated = draw_detections(frame, detections, behavior_dicts)
            path = os.path.join(self.output_dir, "annotated", f"frame_{frame_index:06d}.jpg")
            save_image(annotated, path)

        self._report.frame_analyses.append(analysis)
        return analysis

    def _analyze_buffer(self, frame_index: int) -> Optional[FrameAnalysis]:
        """
        级联模式 — 缓冲区分析（追踪模式）。
        """
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

        person_keyframes = self.extractor.extract_multi_person_keyframes(
            list(self._frame_buffer),
            tracker_enabled=self.tracking_enabled,
        )

        has_tracking = self.tracking_enabled and any(
            d.track_id is not None for d in last_detections
        )

        classify_tasks: list[tuple[int, list[str]]] = []
        task_det_map: dict[int, int] = {}

        for det_idx, det in enumerate(last_detections):
            if has_tracking:
                person_key = det.track_id
            else:
                person_key = det_idx

            keyframes_b64 = person_keyframes.get(person_key)
            if not keyframes_b64:
                continue

            task_idx = len(classify_tasks)
            classify_tasks.append((person_key, keyframes_b64))
            task_det_map[task_idx] = det_idx

        if not classify_tasks:
            return analysis

        tagged_behaviors = self._classify_concurrent(classify_tasks)

        behaviors = []
        behavior_dicts = []

        for task_idx, (person_key, result) in enumerate(tagged_behaviors):
            det_idx = task_det_map.get(task_idx, task_idx)
            person_label = f"track#{person_key}" if has_tracking else f"#{person_key}"

            logger.info(
                f"[帧 {frame_index}] 人物{person_label}: "
                f"{result.behavior_label} ({result.behavior_id}) "
                f"[{result.severity.value}]"
            )

            if self.save_crops and det_idx < len(last_detections):
                self._save_crop(frame, last_detections[det_idx], frame_index, person_key)

            if result.is_alert():
                self._handle_alert(result, frame_index, person_key)

            self._record_behavior(result, frame_index)

            if self._camera_log is not None:
                self._camera_log.add_entry(
                    frame_index=frame_index,
                    person_idx=person_key,
                    result=result,
                )

            behaviors.append(result)
            behavior_dicts.append({
                "person_key": person_key,
                "behavior_label": result.behavior_label,
                "severity": result.severity.value,
            })

        analysis.behaviors = behaviors
        analysis.behavior_dicts = behavior_dicts
        analysis.processing_time = time.time() - start_time

        if self.save_annotated:
            annotated = draw_detections(frame, last_detections, behavior_dicts)
            path = os.path.join(self.output_dir, "annotated", f"frame_{frame_index:06d}.jpg")
            save_image(annotated, path)

        self._report.frame_analyses.append(analysis)
        return analysis

    # ==================================================================
    # 可视化与显示（功能3+功能4）
    # ==================================================================

    def _render_display(
        self,
        frame: np.ndarray,
        detections: list[PersonDetection],
        analysis: Optional[FrameAnalysis] = None,
    ):
        """
        功能3+功能4：渲染显示画面。

        功能4：display_input 和 display_output 控制是否显示输入/输出画面。
        - 只有 output：显示标注后的帧
        - 只有 input：显示原始帧
        - 两者都开启：并排显示（左=原始输入，右=标注输出）
        """
        show_input = self.display_input
        show_output = self.display_output

        # 如果都不显示，只做按键检测
        if not show_input and not show_output:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
            return

        # 构建输出帧
        if show_output:
            annotated = self._draw_frame(frame, detections, analysis)
        else:
            annotated = None

        # 构建输入帧
        if show_input:
            input_frame = frame.copy()
            # 在输入帧上显示简要信息
            info = f"Input | Frame: {self._frame_count}"
            cv2.putText(input_frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(input_frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 组合显示
        if show_input and show_output and annotated is not None:
            # 并排显示
            ih, iw = input_frame.shape[:2]
            oh, ow = annotated.shape[:2]
            # 统一高度
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

        # 缩放
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

        # 添加帧信息
        info = f"Frame: {self._frame_count} | Persons: {len(detections)} | Processed: {self._processed_count}"
        cv2.putText(annotated, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 添加持续帧信息
        cv2.putText(
            annotated,
            f"Sustained: {self._consecutive_detection_count}/{self.sustained_detection_frames}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1,
        )

        # 添加追踪状态
        track_label = "Track: ON" if self.tracking_enabled else "Track: OFF"
        cv2.putText(
            annotated, track_label, (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 200, 200) if self.tracking_enabled else (128, 128, 128), 1,
        )

        # 功能3：在画面上叠加 FPS 信息
        yolo_stats = self.detector.get_fps_stats()
        qwen_stats = self._qwen_fps_tracker.get_stats()
        fps_text = f"YOLO: {yolo_stats['fps']:.1f} FPS ({yolo_stats['avg_ms']:.1f}ms) | Qwen: {qwen_stats['fps']:.1f} FPS ({qwen_stats['avg_ms']:.1f}ms)"
        cv2.putText(annotated, fps_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 添加模式信息
        mode_text = f"Mode: {'Concurrent' if self.concurrent_mode else 'Cascade'} | YOLO skip: {self.detector.yolo_skip_frames}"
        cv2.putText(annotated, mode_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # 添加告警指示
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
    # 功能3：FPS 打印
    # ==================================================================

    def _print_fps_stats(self):
        """功能3：在终端和日志中打印 YOLO 和 Qwen 推理速度"""
        yolo_stats = self.detector.get_fps_stats()
        qwen_stats = self._qwen_fps_tracker.get_stats()

        # 终端打印（带颜色分隔）
        fps_line = (
            f"[FPS] YOLO: {yolo_stats['fps']:6.1f} FPS "
            f"(avg {yolo_stats['avg_ms']:6.1f}ms, count={yolo_stats['count']}) | "
            f"Qwen: {qwen_stats['fps']:6.1f} FPS "
            f"(avg {qwen_stats['avg_ms']:6.1f}ms, count={qwen_stats['count']})"
        )
        logger.info(fps_line)

    # ==================================================================
    # 辅助方法
    # ==================================================================

    def _save_crop(
        self,
        frame: np.ndarray,
        detection: PersonDetection,
        frame_index: int,
        person_idx: int,
        crop_image: Optional[np.ndarray] = None,
    ):
        """保存人体裁剪图。如果传入 crop_image 则直接保存，不再重复裁剪。"""
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
            f"🚨 告警! [帧 {frame_index}] 人物#{person_idx}: "
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

    def _record_behavior(self, result: BehaviorResult, frame_index: int):
        """记录行为到报告统计"""
        bid = result.behavior_id
        self._report.behavior_counts[bid] = self._report.behavior_counts.get(bid, 0) + 1

    def _finalize(self):
        """流水线结束，保存报告"""
        self._report.end_time = time.time()
        self._report.processed_frames = self._processed_count

        # 停止外层并发消费者
        if self.concurrent_mode:
            if hasattr(self, '_consumer_stop_event'):
                self._consumer_stop_event.set()
            if hasattr(self, '_consumer_thread') and self._consumer_thread.is_alive():
                self._consumer_thread.join(timeout=5.0)
                logger.info("外层并发消费者线程已停止")

        # 等待外层并发队列中的剩余任务完成（排空队列）
        if self._concurrent_executor is not None:
            # 先等待队列中已提交的任务完成
            try:
                remaining = self._pending_task_queue.qsize()
                if remaining > 0:
                    logger.info(f"等待 {remaining} 个待处理并发任务完成...")
                    timeout_total = 120  # 最多等120秒
                    start_wait = time.time()
                    while not self._pending_task_queue.empty() and (time.time() - start_wait) < timeout_total:
                        time.sleep(0.5)
            except Exception:
                pass
            try:
                self._concurrent_executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                self._concurrent_executor.shutdown(wait=True)
            logger.info("外层并发线程池已关闭")

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

        # 关闭显示窗口
        if self.display:
            cv2.destroyAllWindows()

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

        # 打印最终摘要（含 FPS 统计）
        summary = self._report.summary()
        yolo_stats = self.detector.get_fps_stats()
        qwen_stats = self._qwen_fps_tracker.get_stats()

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
        logger.info(f"  YOLO 推理速度: {yolo_stats['fps']:.1f} FPS (avg {yolo_stats['avg_ms']:.1f}ms, {yolo_stats['count']} 次)")
        logger.info(f"  Qwen 推理速度: {qwen_stats['fps']:.1f} FPS (avg {qwen_stats['avg_ms']:.1f}ms, {qwen_stats['count']} 次)")
        if self._camera_log is not None:
            logger.info(f"  日志条目: {self._camera_log.entry_count}")
        logger.info("=" * 60)
