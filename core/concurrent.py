"""外层并发模式 — 任务封装与队列管理"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from typing import Optional, Callable

import numpy as np

from models.schemas import PersonDetection, BehaviorResult
from utils.logger import get_logger

logger = get_logger()


class ConcurrentTask:
    """
    外层并发任务封装。

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

        self.tagged_behaviors: list[tuple[int, BehaviorResult]] = []
        self.completed = False
        self.error: Optional[Exception] = None
        self.elapsed: float = 0.0


class ConcurrentQueue:
    """
    并发模式的队列管理器。

    职责：
    - 管理待处理任务队列
    - 管理已完成结果的有序出队
    - 封装消费者线程和外层线程池
    """

    def __init__(
        self,
        execute_fn: Callable[[ConcurrentTask], None],
        max_queued_frames: int = 50,
        max_workers: int = 4,
    ):
        """
        Args:
            execute_fn: 任务执行回调（在工作线程中运行，负责 Qwen 推理）
            max_queued_frames: 最大队列长度
            max_workers: 外层线程池大小
        """
        self._execute_fn = execute_fn
        self._pending_queue: Queue[ConcurrentTask] = Queue(maxsize=max_queued_frames)
        self._completed_results: dict[int, ConcurrentTask] = {}
        self._next_display_task_id: int = 0
        self._task_id_counter: int = 0
        self._lock = threading.Lock()

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="concurrent-qwen",
        )
        self._consumer_stop_event = threading.Event()
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            daemon=True,
            name="concurrent-consumer",
        )

    def start_consumer(self):
        """启动消费者线程"""
        self._consumer_thread.start()

    def next_task_id(self) -> int:
        """获取并递增任务 ID"""
        with self._lock:
            tid = self._task_id_counter
            self._task_id_counter += 1
            return tid

    def submit_task(self, task: ConcurrentTask):
        """将任务入队"""
        try:
            self._pending_queue.put_nowait(task)
            logger.debug(f"并发任务入队: task_id={task.task_id}, frame={task.frame_index}")
        except Exception:
            logger.warning(f"并发队列已满，丢弃帧 {task.frame_index} 的任务")

    def try_dequeue_completed(self) -> Optional[ConcurrentTask]:
        """按帧序号顺序出队已完成的任务"""
        with self._lock:
            task = self._completed_results.get(self._next_display_task_id)
            if task is not None and task.completed:
                del self._completed_results[self._next_display_task_id]
                self._next_display_task_id += 1
                return task
        return None

    def _consumer_loop(self):
        """消费者循环：从队列取出任务提交到线程池"""
        while not self._consumer_stop_event.is_set():
            try:
                task = self._pending_queue.get(timeout=0.5)
            except Empty:
                continue
            self._executor.submit(self._run_task, task)

    def _run_task(self, task: ConcurrentTask):
        """在工作线程中执行任务"""
        try:
            qwen_start = time.time()
            self._execute_fn(task)
            task.elapsed = time.time() - qwen_start
            task.completed = True
        except Exception as e:
            logger.error(f"并发任务执行失败 (task_id={task.task_id}): {e}")
            task.error = e
            task.completed = True

        with self._lock:
            self._completed_results[task.task_id] = task

    def shutdown(self, timeout: float = 5.0):
        """停止消费者并关闭线程池"""
        self._consumer_stop_event.set()
        if self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=timeout)
            logger.info("外层并发消费者线程已停止")

        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=True)
        logger.info("外层并发线程池已关闭")
