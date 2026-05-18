"""视频输入源管理 — 支持文件、USB 摄像头、RTSP 流"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional, Generator

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger()


class VideoSourceType(str, Enum):
    """输入源类型"""
    VIDEO_FILE = "video"
    CAMERA_USB = "camera"
    CAMERA_RTSP = "rtsp"


class VideoSource:
    """
    统一的视频输入源管理器。

    支持：
    - 视频文件（MP4, AVI, MOV 等）
    - USB 摄像头 (cv2.VideoCapture(device_id))
    - RTSP 网络摄像头流
    - 自动断线重连（连续 N 帧失败后触发）
    """

    def __init__(
        self,
        source_type: VideoSourceType,
        input_path: str = "",
        camera_id: int = 0,
        rtsp_url: str = "",
        frame_width: int = 640,
        frame_height: int = 480,
        reconnect_threshold: int = 10,
        reconnect_delay: float = 2.0,
    ):
        """
        Args:
            source_type: 输入源类型
            input_path: 视频文件路径（source_type=VIDEO_FILE 时使用）
            camera_id: USB 摄像头 ID（source_type=CAMERA_USB 时使用）
            rtsp_url: RTSP 流地址（source_type=CAMERA_RTSP 时使用）
            frame_width: 目标帧宽（0=保持原始）
            frame_height: 目标帧高（0=保持原始）
            reconnect_threshold: 连续失败帧数触发重连
            reconnect_delay: 重连等待秒数
        """
        self.source_type = source_type
        self.input_path = input_path
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.target_width = frame_width
        self.target_height = frame_height
        self.reconnect_threshold = reconnect_threshold
        self.reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0
        self._total_frames = 0
        self._frame_index = 0
        self._is_opened = False

    def _get_source_url(self) -> str | int:
        """获取 OpenCV VideoCapture 的输入源"""
        if self.source_type == VideoSourceType.VIDEO_FILE:
            return self.input_path
        elif self.source_type == VideoSourceType.CAMERA_USB:
            return self.camera_id
        elif self.source_type == VideoSourceType.CAMERA_RTSP:
            return self.rtsp_url
        else:
            raise ValueError(f"未知输入源类型: {self.source_type}")

    def open(self) -> bool:
        """
        打开视频源。

        Returns:
            是否成功打开
        """
        source = self._get_source_url()
        logger.info(f"打开视频源: type={self.source_type.value}, source={source}")

        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            self._is_opened = False
            return False

        # 设置目标分辨率
        if self.target_width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        if self.target_height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        # 获取视频信息
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"视频源已打开: {w}x{h} @ {fps:.1f}fps, total_frames={self._total_frames}")
        self._is_opened = True
        self._frame_index = 0
        self._consecutive_failures = 0
        return True

    def _reconnect(self) -> bool:
        """
        重新连接视频源。

        仅对摄像头/RTSP 流有效。视频文件不需要重连。

        Returns:
            是否重连成功
        """
        if self.source_type == VideoSourceType.VIDEO_FILE:
            logger.warning("视频文件不需要重连")
            return False

        logger.warning(
            f"触发重连 (连续 {self._consecutive_failures} 帧失败), "
            f"等待 {self.reconnect_delay}s..."
        )

        # 释放旧资源
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        time.sleep(self.reconnect_delay)

        # 重新打开
        source = self._get_source_url()
        self._cap = cv2.VideoCapture(source)

        if self._cap.isOpened():
            # 恢复分辨率设置
            if self.target_width > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            if self.target_height > 0:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

            logger.info("重连成功")
            self._consecutive_failures = 0
            return True
        else:
            logger.error("重连失败")
            self._cap = None
            return False

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧。

        Returns:
            (success, frame): success=False 且 frame=None 表示结束或错误
        """
        if self._cap is None or not self._cap.isOpened():
            if not self.open():
                return False, None

        ret, frame = self._cap.read()

        if ret and frame is not None:
            self._consecutive_failures = 0
            self._frame_index += 1
            return True, frame

        # 读取失败
        self._consecutive_failures += 1

        # 视频文件结束
        if self.source_type == VideoSourceType.VIDEO_FILE:
            logger.info(f"视频播放完毕, 共 {self._frame_index} 帧")
            return False, None

        # 摄像头/RTSP: 检查是否需要重连
        if self._consecutive_failures >= self.reconnect_threshold:
            logger.warning(f"连续 {self._consecutive_failures} 帧读取失败，尝试重连...")
            self._reconnect()
            # 重连后返回 False，由 frames() 循环继续重试

        return False, None

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        帧迭代器，自动处理重连。

        Yields:
            BGR 格式的帧图像
        """
        while True:
            ret, frame = self.read()
            if not ret:
                # 对于实时摄像头，重连失败后继续尝试
                if self.source_type in (VideoSourceType.CAMERA_USB, VideoSourceType.CAMERA_RTSP):
                    time.sleep(0.1)
                    continue
                else:
                    break
            yield frame

    def release(self):
        """释放视频源"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False
        logger.info("视频源已释放")

    @property
    def is_opened(self) -> bool:
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def frame_size(self) -> tuple[int, int]:
        """当前帧尺寸 (width, height)"""
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def __del__(self):
        self.release()
