# 智能救生行为识别 Agent

> 基于 **YOLOv8/v11 人体检测** + **千问多模态大模型** 的实时水域安全行为分析系统

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [环境准备](#环境准备)
- [安装依赖](#安装依赖)
- [配置说明](#配置说明)
- [运行方式](#运行方式)
- [本地模型部署](#本地模型部署可选)
- [微调 YOLO](#微调-yolo可选)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 项目简介

本系统实现了从摄像头/视频输入到行为识别告警的完整流水线：

```
摄像头/视频 → YOLO 人体检测 → 裁剪人体区域关键帧 → 多模态大模型分析 → 输出结果/告警
```

**核心特性：**
- 🎯 YOLOv8/v11 人体检测 + ByteTrack 目标跟踪
- 🧠 云端千问 API（qwen3-vl-flash）/ 本地 vLLM 部署双模式推理
- 🔍 自适应 padding + 关键帧提取
- 📊 可扩展行为类别（溺水/游泳/翻栏杆/正常行走/救援/在船上）
- ⚠️ 分级告警（critical / warning / normal）+ 冷却机制
- 📝 摄像头行为日志（自动清理过期记录）
- 🔄 断线重连（RTSP/USB 摄像头）

---

## 系统架构

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌─────────────┐
│  VideoSource │───>│  Detector    │───>│ FrameExtractor   │───>│ Classifier  │
│  视频/摄像头  │    │  YOLO+Track  │    │ 裁剪+自适应padding│    │ 千问多模态  │
└──────────────┘    └──────────────┘    └──────────────────┘    └─────────────┘
                                                                       │
                                                                       v
                                                              ┌──────────────┐
                                                              │   Pipeline   │
                                                              │ 告警/日志/显示 │
                                                              └──────────────┘
```

---

## 环境准备

### 系统要求

| 项目 | 最低要求 |
|------|---------|
| 操作系统 | Linux / Windows / macOS |
| Python | 3.10 ~ 3.12 |
| GPU（可选） | NVIDIA + CUDA 11.8+，没有也能跑 CPU |

### 安装 Python

**Linux (Ubuntu/Debian)：**
```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv -y
```

**Windows：**
前往 [python.org](https://www.python.org/downloads/) 下载安装，勾选 **Add to PATH**。

**macOS：**
```bash
brew install python
```

### 创建虚拟环境

**方式 A：conda（推荐）**

```bash
# 安装 Miniconda（如未安装）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建环境
conda create -n agent python=3.10 -y
conda activate agent
```

**方式 B：venv**

```bash
python3 -m venv agent-env
source agent-env/bin/activate    # Linux/macOS
# agent-env\Scripts\activate     # Windows
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

国内镜像加速：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

| 包 | 用途 |
|---|---|
| `ultralytics` | YOLOv8/v11 人体检测 + ByteTrack/BotSORT 跟踪 |
| `opencv-python` | 视频采集、图像处理、RTSP/USB 摄像头 |
| `openai` | 调用千问 API（OpenAI 兼容接口） |
| `PyYAML` | 配置文件管理 |
| `loguru` | 日志输出 |

---

## 配置说明

编辑 `config.yaml`，关键配置项如下：

### 模型运行模式

```yaml
model_mode: "local"    # "api" = 阿里云百炼千问 | "local" = 本地 vLLM
```

### 提示词模式

```yaml
prompt_mode: "detailed"  # "detailed" = 详细版 | "brief" = 简练版（省 token）
```

提示词模板存放于 `prompts/` 目录，支持自定义编辑。

### 云端 API 配置（model_mode = "api"）

```yaml
qwen:
  api_key: ""                    # 也可通过环境变量 QWEN_API_KEY 设置
  api_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen3-vl-flash"       # 性价比高；可选 "qwen-vl-max"（更准但更贵）
  max_tokens: 100
  temperature: 0.01
```

### 本地 vLLM 配置（model_mode = "local"）

```yaml
local_model:
  api_key: "abc123"
  api_url: "http://localhost:7890/v1"
  model: "Qwen/Qwen3-VL-4B-AWQ"
  max_tokens: 128
  temperature: 0.1
  timeout: 60
```

### 检测器 (detector)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `yolo11n.pt` | YOLO 模型路径，首次运行自动下载 |
| `confidence` | `0.1` | 检测置信度阈值 |
| `device` | `cuda:0` | 推理设备：`cpu` / `cuda:0` |
| `class_ids` | `[0, 1, 2]` | 检测类别 ID 列表，全部当作"人"处理 |
| `detect_width` | `640` | 推理宽度，`0` = 保持原始分辨率 |
| `detect_height` | `640` | 推理高度，`0` = 保持原始分辨率 |
| `nms_iou` | `0.1` | NMS IoU 阈值，越小越严格 |

### 跟踪器 (tracker)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `false` | `true` = 启用跟踪（带 ID）/ `false` = 仅检测 |
| `tracker_type` | `bytetrack` | `bytetrack` / `botsort` |
| `track_high_thresh` | `0.6` | 高置信度跟踪阈值 |
| `track_low_thresh` | `0.3` | 低置信度跟踪阈值 |
| `match_thresh` | `0.8` | IoU 匹配阈值 |
| `track_buffer` | `30` | 跟踪丢失后保留帧数 |

### 关键帧提取 (frame_extractor)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `padding_ratio` | `0.15` | 裁剪 padding 倍率 |
| `keyframe_interval` | `1` | 每隔几帧提取一帧 |
| `keyframe_count` | `1` | 关键帧数量上限 |
| `adaptive_padding` | `true` | 小目标自动放大 padding |
| `pixel_threshold` | `20000` | 最小裁剪面积阈值（像素²） |

### 行为类别

| ID | 中文 | 英文 | 严重等级 |
|----|------|------|---------|
| 0 | 溺水 | drowning | 🔴 critical |
| 1 | 游泳 | swimming | 🟢 normal |
| 2 | 攀爬栏杆 | climbing | 🟠 warning |
| 3 | 正常行走 | normal_walking | 🟢 normal |
| 4 | 正在救援 | waterhelping | 🟢 normal |
| 5 | 在船上 | abord | 🟢 normal |

> 行为类别可在 `config.yaml` 中的 `behavior_classes` 自定义扩展。

### 流水线 (pipeline)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `process_every_n_frames` | `1` | 每 N 帧触发一次推理 |
| `buffer_size` | `1` | 历史帧缓冲区大小（仅追踪模式） |
| `max_concurrent` | `5` | 最大并发 API 请求数 |
| `alert_cooldown` | `5` | 同一行为告警冷却（秒） |
| `display` | `true` | 是否显示实时画面 |
| `display_scale` | `0.1` | 视频窗口缩放比例 |
| `sustained_detection_frames` | `1` | 连续 N 帧检测到目标才触发 API |

---

## 运行方式

### 1. 云端 API 分析视频

需要先获取 [阿里云百炼](https://bailian.console.aliyun.com/) API Key。

```bash
# 方式 A：写入 config.yaml
# qwen:
#   api_key: "sk-xxx"

# 方式 B：环境变量
export QWEN_API_KEY="sk-xxx"

# 方式 C：命令行参数
python main.py --source video --input video.mp4 --api-key sk-xxx --no-display
```

### 2. 本地 vLLM 分析视频

```bash
# 确保 vLLM 服务已启动（见下方）
python main.py --source video --input video.mp4 --model-mode local --no-display
```

### 3. USB 摄像头

```bash
python main.py --source camera --camera-id 0 --model-mode local

# Linux 查看摄像头设备
ls /dev/video*
```

### 4. RTSP 网络摄像头

```bash
python main.py --source rtsp --rtsp-url rtsp://192.168.1.100:554/stream --model-mode local --no-display

# 测试 RTSP 是否可用
ffplay rtsp://192.168.1.100:554/stream
```

### 5. 完整命令行参数

```bash
python main.py \
  --source video \
  --input /path/to/video.mp4 \
  --model-mode local \
  --no-display \
  --config config.yaml \
  --output output/ \
  --display-scale 0.5 \
  --max-concurrent 5
```

| 参数 | 说明 |
|------|------|
| `--source, -s` | 输入源：`video` / `camera` / `rtsp` |
| `--input, -i` | 视频文件路径 |
| `--camera-id` | USB 摄像头 ID（默认 0） |
| `--rtsp-url` | RTSP 流地址 |
| `--model-mode` | 模型模式：`api` / `local`（优先于 config） |
| `--api-key` | 千问 API Key |
| `--config, -c` | 配置文件路径（默认 config.yaml） |
| `--output, -o` | 输出目录 |
| `--no-display` | 无头模式（服务器环境） |
| `--no-crops` | 不保存人体裁剪图 |
| `--display-scale` | 窗口缩放比例 |
| `--max-concurrent` | 最大并发数 |
| `--verbose, -v` | 详细日志 |

---

## 本地模型部署（可选）

不需要 API Key，完全本地运行，数据不出本机。

### 1. 安装 vLLM

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm
```

### 2. 下载模型

> 需查看 [vLLM 官方支持模型列表](https://docs.vllm.com.cn/en/latest/usage/)

```bash
pip install modelscope

# Qwen3-VL-4B-Instruct (FP16, ~8GB)
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir /your/path/qwen3-vl-4b

# Qwen3-VL-4B-Instruct-AWQ-4bit (INT4 量化, ~4GB)
python -c "
from modelscope import snapshot_download
snapshot_download('cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit', local_dir='/your/path/qwen3-vl-4b-awq')
"

# Qwen3.5-2B (轻量版)
modelscope download --model Qwen/Qwen3.5-2B --local_dir /your/path/qwen3.5-2b
```

### 3. 启动 vLLM 服务

```bash
# 单卡启动
CUDA_VISIBLE_DEVICES=1 vllm serve /your/path/qwen3-vl-4b \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --max-model-len 1024 \
  --port 7890 \
  --gpu-memory-utilization 0.25

# 多卡张量并行
CUDA_VISIBLE_DEVICES=0,1 vllm serve /your/path/qwen3-vl-4b \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --max-model-len 4096 \
  --tensor-parallel-size 2 \
  --port 7890
```

| 参数 | 说明 |
|------|------|
| `--api-key` | 自定义密钥，客户端请求需一致 |
| `--served-model-name` | 模型名称标识，`config.yaml` 中需对应 |
| `--max-model-len` | 最大 token 总数（输入+输出） |
| `--port` | 服务端口 |
| `--gpu-memory-utilization` | GPU 显存占用比例（0.25 = 25%） |
| `--max-num-seqs` | 最大并发序列数 |

看到 `Uvicorn running on http://0.0.0.0:7890` 表示启动成功。

### 4. 验证服务

```bash
curl http://127.0.0.1:7890/v1/models -H "Authorization: Bearer abc123"
```

### 5. 切换 config.yaml

```yaml
model_mode: "local"

local_model:
  api_key: "abc123"                    # 与 --api-key 一致
  api_url: "http://localhost:7890/v1"
  model: "Qwen/Qwen3-VL-4B-Instruct"  # 与 --served-model-name 一致
```

---

## 微调 YOLO（可选）

如需在特定场景（如泳池）提升检测精度，可用自有数据集微调。

### 准备数据集

```
dataset/
├── train/
│   ├── images/     # 训练图片
│   └── labels/     # 标注 .txt 文件
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml    # 数据集配置
```

`dataset.yaml` 示例：
```yaml
path: ./dataset
train: train/images
val: val/images
names:
  0: person
  1: drowning
```

### 开始训练

```bash
python finetune_yolo.py --data dataset.yaml --pretrained yolov8n.pt --epochs 100 --lr 0.001
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集配置文件 | 必填 |
| `--pretrained` | 预训练权重 | `yolov8n.pt` |
| `--epochs` | 训练轮数 | `50` |
| `--lr` | 学习率 | `0.001` |
| `--batch` | 批次大小 | `16` |

### 使用微调模型

训练完成后，最佳权重位于：
```
runs/finetune/finetune_xxxx/weights/best.pt
```

更新 `config.yaml`：
```yaml
detector:
  model: "runs/finetune/finetune_xxxx/weights/best.pt"
```

---

## 项目结构

```
actionv3/
├── main.py                        # 主入口
├── config.yaml                    # 全局配置
├── prompts/                       # 提示词模板
│   ├── detailed_prompt.txt        #   详细版提示词
│   └── brief_prompt.txt           #   简练版提示词
├── core/
│   ├── __init__.py
│   ├── detector.py                #   YOLO 人体检测 + ByteTrack/BotSORT 跟踪
│   ├── pipeline.py                #   推理流水线（检测→提取→分类→告警→日志）
│   ├── behavior_classifier.py     #   行为分类器（云端 API / 本地 vLLM）
│   ├── frame_extractor.py         #   关键帧提取 + 自适应 padding
│   └── video_source.py            #   视频源管理（文件/USB/RTSP + 断线重连）
├── models/
│   ├── __init__.py
│   └── schemas.py                 #   数据模型（BoundingBox, Detection, BehaviorResult...）
├── utils/
│   ├── __init__.py
│   ├── logger.py                  #   loguru 日志配置
│   └── image_utils.py             #   图像工具（裁剪、编码、标注、自适应 padding）
├── finetune_yolo.py               # YOLO 微调脚本
├── tracker_guide.md               # 跟踪器调参指南
├── DEPLOYMENT.md                  # 部署文档
└── requirements.txt               # 依赖列表
```

---

## 输出文件

运行结束后在 `output/` 目录下生成：

```
output/
├── analysis_report.json           # 分析报告（帧数、行为统计、运行时长）
├── alerts.json                    # 告警记录
├── camera_behavior_log.json       # 摄像头行为日志（自动清理过期条目）
├── annotated/                     # 标注后的帧图片
└── crops/                         # 人体裁剪图
```

---

## 常见问题

### Q: pip install 报错
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: YOLO 模型下载失败
手动下载放入项目目录：
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
```

### Q: 摄像头打不开
```bash
ls /dev/video*                    # 查看设备
sudo usermod -aG video $USER      # 加权限（需重新登录）
```

### Q: CUDA 报错 / 没有 GPU
```yaml
detector:
  device: "cpu"
```

### Q: API 调用失败
- 检查 API Key 是否正确
- 检查网络是否能访问 `dashscope.aliyuncs.com`
- 检查阿里云账户余额

### Q: 行为识别不准
- 增加 `keyframe_count`（如 `3`），给模型更多信息
- 使用更好的模型：`model: "qwen-vl-max"`
- 切换为详细提示词：`prompt_mode: "detailed"`

### Q: 运行太慢
- 降低 `detect_width` / `detect_height`（如 `640` → `320`）
- 增大 `process_every_n_frames`（如 `5`）
- 使用 GPU：`device: "cuda:0"`
- 使用量化模型（AWQ-4bit）减少显存占用
