"""
智能救生行为识别 Agent — 主入口

支持两种模型模式：
    - api:   云端千问 API（默认）
    - local: 本地 vLLM 部署的模型

用法：
    # 云端 API 分析视频
    python main.py --source video --input /path/to/video.mp4

    # 本地 vLLM 模型分析视频
    python main.py --source video --input /path/to/video.mp4 --model-mode local

    # USB 摄像头
    python main.py --source camera --camera-id 0

    # RTSP 流
    python main.py --source rtsp --rtsp-url rtsp://192.168.1.100:554/stream

环境变量：
    QWEN_API_KEY    千问 API 密钥（云端模式必填）
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

from core.detector import PersonDetector
from core.frame_extractor import FrameExtractor
from core.behavior_classifier import BehaviorClassifier
from core.video_source import VideoSource, VideoSourceType
from core.pipeline import Pipeline
from utils.logger import setup_logger, get_logger


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    if not os.path.exists(config_path):
        print(f"警告: 配置文件不存在: {config_path}，使用默认配置")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def build_pipeline(args: argparse.Namespace, config: dict) -> Pipeline:
    """
    根据命令行参数和配置文件构建流水线。

    优先级：命令行参数 > 配置文件 > 默认值
    """
    # ---- 日志 ----
    logger = setup_logger(level="INFO")

    # ---- 视频源 ----
    source_type = VideoSourceType(args.source)

    vs_cfg = config.get("video_source", {})
    video_source = VideoSource(
        source_type=source_type,
        input_path=args.input or "",
        camera_id=args.camera_id if args.camera_id is not None else vs_cfg.get("camera_id", 0),
        rtsp_url=args.rtsp_url or vs_cfg.get("rtsp_url", ""),
        frame_width=vs_cfg.get("frame_width", 640),
        frame_height=vs_cfg.get("frame_height", 480),
        reconnect_threshold=vs_cfg.get("reconnect_threshold", 10),
        reconnect_delay=vs_cfg.get("reconnect_delay", 2.0),
    )

    # ---- 人体检测器 ----
    det_cfg = config.get("detector", {})
    trk_cfg = config.get("tracker", {})
    detector = PersonDetector(
        model_path=det_cfg.get("model", "yolov8n.pt"),
        confidence=det_cfg.get("confidence", 0.5),
        device=det_cfg.get("device", "cpu"),
        class_ids=det_cfg.get("class_ids", det_cfg.get("class_id", 0)),
        detect_width=det_cfg.get("detect_width", 0),
        detect_height=det_cfg.get("detect_height", 0),
        tracker_enabled=trk_cfg.get("enabled", True),
        tracker_type=trk_cfg.get("tracker_type", "bytetrack"),
        track_high_thresh=trk_cfg.get("track_high_thresh", 0.5),
        track_low_thresh=trk_cfg.get("track_low_thresh", 0.1),
        match_thresh=trk_cfg.get("match_thresh", 0.8),
        track_buffer=trk_cfg.get("track_buffer", 30),
        nms_iou=det_cfg.get("nms_iou", 0.5),
        with_reid=trk_cfg.get("with_reid", False),
    )

    # ---- 关键帧提取器 ----
    fe_cfg = config.get("frame_extractor", {})
    frame_extractor = FrameExtractor(
        padding_ratio=fe_cfg.get("padding_ratio", 0.15),
        keyframe_interval=fe_cfg.get("keyframe_interval", 3),
        keyframe_count=fe_cfg.get("keyframe_count", 5),
        min_region_size=fe_cfg.get("min_region_size", 32),
        adaptive_padding=fe_cfg.get("adaptive_padding", True),
        pixel_threshold=fe_cfg.get("pixel_threshold", 10000.0),
    )

    # ---- 行为分类器 ----
    behavior_classes = config.get("behavior_classes", None)
    model_mode = args.model_mode or config.get("model_mode", "api")
    prompt_mode = config.get("prompt_mode", "detailed")
    
    if model_mode == "local":
        lm_cfg = config.get("local_model", {})
        classifier = BehaviorClassifier(
            api_key=args.api_key or lm_cfg.get("api_key", ""),
            api_url=lm_cfg.get("api_url", "http://localhost:7890/v1"),
            model=lm_cfg.get("model", "Qwen/Qwen3-VL-4B-Instruct"),
            max_tokens=lm_cfg.get("max_tokens", 512),
            temperature=lm_cfg.get("temperature", 0.1),
            timeout=lm_cfg.get("timeout", 60),
            behavior_classes=behavior_classes,
            model_mode="local",
            prompt_mode=prompt_mode,
        )
    else:
        qw_cfg = config.get("qwen", {})
        classifier = BehaviorClassifier(
            api_key=args.api_key or qw_cfg.get("api_key", ""),
            api_url=qw_cfg.get("api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=qw_cfg.get("model", "qwen3-vl-flash"),
            max_tokens=qw_cfg.get("max_tokens", 512),
            temperature=qw_cfg.get("temperature", 0.1),
            timeout=qw_cfg.get("timeout", 30),
            behavior_classes=behavior_classes,
            model_mode="api",
            prompt_mode=prompt_mode,
        )

    # ---- 流水线 ----
    pp_cfg = config.get("pipeline", {})
    out_cfg = config.get("output", {})

    output_dir = args.output or out_cfg.get("output_dir", "output")

    pipeline = Pipeline(
        detector=detector,
        frame_extractor=frame_extractor,
        classifier=classifier,
        video_source=video_source,
        process_every_n_frames=pp_cfg.get("process_every_n_frames", 5),
        buffer_size=pp_cfg.get("buffer_size", 5),
        camera_interval=pp_cfg.get("camera_interval", 0.1),
        alert_cooldown=pp_cfg.get("alert_cooldown", 30),
        sustained_detection_frames=pp_cfg.get("sustained_detection_frames", 1),  # 需求3
        max_concurrent=pp_cfg.get("max_concurrent", 1),  # 最大并发 API 请求数
        output_dir=output_dir,
        save_annotated=out_cfg.get("save_annotated", True),
        save_crops=out_cfg.get("save_crops", True),
        save_report=out_cfg.get("save_report", True),
        display=pp_cfg.get("display", True),
        display_scale=pp_cfg.get("display_scale", 0.5),
        # 需求2：摄像头日志配置
        camera_log_enabled=config.get("camera_log", {}).get("enabled", True),
        camera_log_retention_hours=config.get("camera_log", {}).get("retention_hours", 2.0),
        camera_log_filename=config.get("camera_log", {}).get("log_filename", "camera_behavior_log.json"),
    )

    return pipeline


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="智能救生行为识别 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用云端 API 分析视频
  %(prog)s --source video --input pool_video.mp4

  # 使用本地 vLLM 模型分析视频
  %(prog)s --source video --input pool_video.mp4 --model-mode local

  # USB 摄像头 + 本地模型
  %(prog)s --source camera --camera-id 0 --model-mode local

  # RTSP 流 + 云端 API
  %(prog)s --source rtsp --rtsp-url rtsp://192.168.1.100:554/stream --api-key sk-xxx --no-display
        """,
    )

    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        choices=["video", "camera", "rtsp"],
        help="输入源类型: video(视频文件) / camera(USB摄像头) / rtsp(网络摄像头)",
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default="",
        help="视频文件路径 (source=video 时必填)",
    )

    parser.add_argument(
        "--camera-id",
        type=int,
        default=None,
        help="USB 摄像头设备 ID (默认: 0)",
    )

    parser.add_argument(
        "--rtsp-url",
        type=str,
        default="",
        help="RTSP 流地址 (source=rtsp 时必填)",
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="千问 API Key (也可通过环境变量 QWEN_API_KEY 设置)",
    )

    parser.add_argument(
        "--model-mode",
        type=str,
        choices=["api", "local"],
        default=None,
        help="模型运行模式: api(云端千问API) / local(本地vLLM部署), 优先级高于config.yaml",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="输出目录 (默认: output/)",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="不显示实时画面（无头模式）",
    )

    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="不保存人体裁剪图",
    )

    parser.add_argument(
        "--camera-interval",
        type=float,
        default=0.1,
        help="摄像头调用间隔（秒）",
    )

    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.5,
        help="视频窗口缩放比例",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="最大并发 API 请求数 (默认: 读取 config.yaml 中的 max_concurrent)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行覆盖
    if args.no_display:
        config.setdefault("pipeline", {})["display"] = False
    if args.no_crops:
        config.setdefault("output", {})["save_crops"] = False
    if args.camera_interval:
        config.setdefault("pipeline", {})["camera_interval"] = args.camera_interval
    if args.display_scale:
        config.setdefault("pipeline", {})["display_scale"] = args.display_scale
    if args.max_concurrent is not None:
        config.setdefault("pipeline", {})["max_concurrent"] = max(1, args.max_concurrent)

    # 设置 API Key（命令行优先，其次环境变量）
    if args.api_key:
        os.environ["QWEN_API_KEY"] = args.api_key

    # 构建流水线
    logger = setup_logger(level="DEBUG" if args.verbose else "INFO")
    logger.info("正在初始化行为识别 Agent...")

    try:
        pipeline = build_pipeline(args, config)
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        sys.exit(1)

    # 运行
    pipeline.run()


if __name__ == "__main__":
    main()
