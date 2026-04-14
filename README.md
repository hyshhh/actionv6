# 智能救生行为识别 Agent

> 基于 **YOLOv8/v11 人体检测** + **千问多模态大模型** 的实时水域安全行为分析系统

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [环境准备](#环境准备)
- [安装依赖](#安装依赖)
- [配置文件详解](#配置文件详解)
- [运行方式](#运行方式)
  - [视频文件运行示例](#6-视频文件运行示例)
- [本地模型部署](#本地模型部署可选)
- [微调 YOLO](#微调-yolo可选)
- [性能监控](#性能监控)
  - [三类指标详解](#三类指标详解)
  - [数据流示意](#数据流示意)
- [项目结构](#项目结构)
- [输出文件](#输出文件)
- [常见问题](#常见问题)

---

## 项目简介

本系统实现了从摄像头/视频输入到行为识别告警的完整流水线：

```
摄像头/视频 → YOLO 人体检测 → 裁剪人体区域关键帧 → 多模态大模型分析 → 输出结果/告警
```

**核心特性：**
- 🎯 YOLOv8/v11 人体检测 + ByteTrack / BotSORT 目标跟踪
- 🧠 云端千问 API / 本地 vLLM 部署双模式推理
- ⚡ YOLO 隔帧推理（节省 GPU 资源）
- 🔀 外层并发模式（YOLO 不等 Qwen，异步出队结果）
- 🔍 自适应 padding + 关键帧提取
- 📊 可扩展行为类别（溺水/游泳/翻栏杆/正常行走/救援/在船上）
- ⚠️ 分级告警（critical / warning / normal）+ 冷却机制
- 📝 摄像头行为日志（自动清理过期记录）
- 🔄 断线重连（RTSP/USB 摄像头）
- 📈 实时速度监控：YOLO FPS、Qwen 吞吐、码流速度

---

## 系统架构

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌─────────────┐
│  VideoSource │───>│  Detector    │───>│ FrameExtractor   │───>│ Classifier  │
│  视频/摄像头  │    │  YOLO+Track  │    │ 裁剪+自适应padding│    │ 千问多模态  │
└──────────────┘    └──────────────┘    └──────────────────┘    └─────────────┘
       │                   │                                           │
       │                   v                                           v
       │            ┌─────────────┐                             ┌──────────────┐
       │            │FrameRate    │                             │   Pipeline   │
       └───────────>│Tracker      │                             │ 告警/日志/显示 │
                    │Bitrate      │                             └──────────────┘
                    │Tracker      │
                    └─────────────┘
```

**两种处理模式：**

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| 级联模式 | 每触发点同步等 Qwen 结果再继续 | 简单、低资源 |
| 并发模式 (`concurrent_mode: true`) | YOLO 不等 Qwen，结果异步出队 | 高吞吐、实时性要求高 |

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

## 配置文件详解

配置文件为 `config.yaml`，以下是每个字段的完整说明。

### 顶层配置

```yaml
model_mode: "local"      # 模型运行模式
prompt_mode: "detailed"  # 提示词模式
```

| 字段 | 类型 | 可选值 | 说明 |
|------|------|--------|------|
| `model_mode` | string | `"api"` / `"local"` | `"api"` = 调用阿里云百炼千问 API；`"local"` = 调用本地 vLLM 部署的模型 |
| `prompt_mode` | string | `"detailed"` / `"brief"` | `"detailed"` = 详细提示词（效果更好，token 消耗更高）；`"brief"` = 简练提示词（轻量运行） |

提示词模板存放于 `prompts/` 目录，支持自定义编辑。

---

### 云端 API 配置（`model_mode = "api"` 时生效）

```yaml
qwen:
  api_key: ""
  api_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen3-vl-flash"
  max_tokens: 100
  temperature: 0.01
  timeout: 30
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | string | `""` | API 密钥。也可通过环境变量 `QWEN_API_KEY` 设置，优先级：命令行 `--api-key` > 环境变量 > config |
| `api_url` | string | 见上 | 阿里云百炼 OpenAI 兼容接口地址，一般不用改 |
| `model` | string | `"qwen3-vl-flash"` | 模型名称。可选 `"qwen-vl-max"`（更准但更贵） |
| `max_tokens` | int | `100` | 最大生成 token 数。行为识别一般 100-512 足够 |
| `temperature` | float | `0.01` | 生成温度。越低越确定，推荐 `0.01` ~ `0.1` |
| `timeout` | int | `30` | API 请求超时（秒） |

---

### 本地 vLLM 配置（`model_mode = "local"` 时生效）

```yaml
local_model:
  api_key: "abc123"
  api_url: "http://localhost:7890/v1"
  model: "Qwen/Qwen3-VL-4B-AWQ"
  max_tokens: 128
  temperature: 0.1
  timeout: 60
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | string | `"abc123"` | vLLM `--api-key` 设置的密钥。无鉴权时留空 |
| `api_url` | string | `"http://localhost:7890/v1"` | vLLM 服务地址 |
| `model` | string | `"Qwen/Qwen3-VL-4B-AWQ"` | `--served-model-name` 设置的模型名，必须一致 |
| `max_tokens` | int | `128` | 最大生成 token 数 |
| `temperature` | float | `0.1` | 生成温度 |
| `timeout` | int | `60` | 本地推理超时（秒），本地推理可能更慢 |

---

### 人体检测器（`detector`）

```yaml
detector:
  model: "yolo11n.pt"
  confidence: 0.1
  device: "cuda:0"
  class_ids: [0, 1, 2]
  detect_width: 640
  detect_height: 640
  nms_iou: 0.1
  yolo_skip_frames: 0
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | `"yolo11n.pt"` | YOLO 模型文件路径。首次运行自动下载。常用：`yolov8n.pt`（快）、`yolov8s.pt`（准）、`yolo11n.pt`（最新） |
| `confidence` | float | `0.1` | 检测置信度阈值。低于此值的检测框会被丢弃。越低召回越多，但误检也越多 |
| `device` | string | `"cuda:0"` | 推理设备。`"cpu"` = 纯 CPU；`"cuda:0"` = 第一块 GPU |
| `class_ids` | list[int] | `[0, 1, 2]` | 要检测的 COCO 类别 ID 列表。`0`=人，`1`=自行车，`2`=汽车。全部当作"人"处理 |
| `detect_width` | int | `640` | 推理输入宽度（像素）。`0` = 保持原始分辨率。降低可加速但损失精度 |
| `detect_height` | int | `640` | 推理输入高度（像素）。`0` = 保持原始分辨率 |
| `nms_iou` | float | `0.1` | NMS（非极大值抑制）IoU 阈值。越小越严格，更多重叠框被合并。推荐 `0.1` ~ `0.5` |
| `yolo_skip_frames` | int | `0` | YOLO 隔帧推理间隔。`0` = 每帧都推理；`N` = 每 N+1 帧推理 1 次，中间帧复用上次结果。节省 GPU 资源 |

---

### 目标跟踪器（`tracker`）

```yaml
tracker:
  enabled: false
  tracker_type: "bytetrack"
  track_high_thresh: 0.6
  track_low_thresh: 0.3
  match_thresh: 0.8
  track_buffer: 30
  with_reid: false
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `false` | 是否启用目标跟踪。`true` = 为每个人分配稳定 ID；`false` = 仅检测，无 ID |
| `tracker_type` | string | `"bytetrack"` | 跟踪器类型。`"bytetrack"` = 速度快；`"botsort"` = 更稳定但更慢 |
| `track_high_thresh` | float | `0.6` | 高置信度跟踪阈值。高于此值的检测框直接匹配 |
| `track_low_thresh` | float | `0.3` | 低置信度跟踪阈值。ByteTrack 二次匹配用 |
| `match_thresh` | float | `0.8` | IoU 匹配阈值。越高要求匹配越精确 |
| `track_buffer` | int | `30` | 跟踪丢失后保留帧数。人暂时离开画面时 ID 不会立刻消失 |
| `with_reid` | bool | `false` | BotSORT 是否启用 ReID 外观特征匹配（需额外下载模型） |

> 详细调参参考 `tracker_guide.md`。

---

### 关键帧提取（`frame_extractor`）

```yaml
frame_extractor:
  padding_ratio: 0.15
  keyframe_interval: 1
  keyframe_count: 1
  min_region_size: 32
  adaptive_padding: true
  pixel_threshold: 20000
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `padding_ratio` | float | `0.15` | 正常 padding 倍率。检测框四周各扩展 15%。大目标使用此值 |
| `keyframe_interval` | int | `1` | 每隔几帧提取一帧关键帧。`1` = 每帧都取 |
| `keyframe_count` | int | `1` | 提取的关键帧数量上限。更多帧给模型更多信息，但 token 消耗更高 |
| `min_region_size` | int | `32` | 最小有效区域像素。裁剪区域小于此值时跳过 |
| `adaptive_padding` | bool | `true` | 是否启用自适应 padding。小目标自动放大裁剪区域 |
| `pixel_threshold` | float | `20000` | 最小裁剪面积阈值（像素²）。检测框面积小于此值时，padding 自动放大使裁剪区域达到此面积 |

---

### 行为类别（`behavior_classes`）

```yaml
behavior_classes:
  - id: "0"
    label_cn: 溺水
    label_en: drowning
    severity: critical
    description: >
      四肢无规律挣扎，有溺水风险。
  # ... 更多类别
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 行为唯一标识，模型返回的 `behavior_id` 必须匹配 |
| `label_cn` | string | 中文标签 |
| `label_en` | string | 英文标签 |
| `severity` | string | 严重等级：`critical`（危险告警）/ `warning`（警告）/ `normal`（正常） |
| `description` | string | 行为描述，会写入提示词帮助模型识别 |

**内置行为类别：**

| ID | 中文 | 英文 | 严重等级 | 说明 |
|----|------|------|---------|------|
| 0 | 溺水 | drowning | 🔴 critical | 四肢无规律挣扎 |
| 1 | 游泳 | swimming | 🟢 normal | 正常游泳 |
| 2 | 攀爬栏杆 | climbing | 🟠 warning | 攀爬或翻越栏杆 |
| 3 | 正常行走 | normal_walking | 🟢 normal | 岸上正常行走或站立 |
| 4 | 正在救援 | waterhelping | 🟢 normal | 水中人员抱住救生圈 |
| 5 | 在船上 | abord | 🟢 normal | 人员在船上或开船 |

> 可在 `behavior_classes` 中自定义扩展新类别。添加后对应的提示词模板也需相应调整。

---

### 视频源（`video_source`）

```yaml
video_source:
  camera_id: 0
  rtsp_url: ""
  frame_width: 640
  frame_height: 480
  reconnect_threshold: 10
  reconnect_delay: 2.0
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `camera_id` | int | `0` | USB 摄像头设备 ID。Linux 下 `ls /dev/video*` 查看 |
| `rtsp_url` | string | `""` | RTSP 流地址，如 `rtsp://192.168.1.100:554/stream` |
| `frame_width` | int | `640` | 目标帧宽度。`0` = 保持原始分辨率 |
| `frame_height` | int | `480` | 目标帧高度。`0` = 保持原始分辨率 |
| `reconnect_threshold` | int | `10` | 连续失败多少帧后触发重连（仅摄像头/RTSP） |
| `reconnect_delay` | float | `2.0` | 重连等待时间（秒） |

---

### 输出配置（`output`）

```yaml
output:
  save_annotated: true
  save_crops: true
  save_report: true
  output_dir: "output"
  report_format: "json"
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_annotated` | bool | `true` | 是否保存标注后的帧图片（带检测框和行为标签） |
| `save_crops` | bool | `true` | 是否保存人体裁剪图 |
| `save_report` | bool | `true` | 是否保存分析报告 JSON |
| `output_dir` | string | `"output"` | 输出目录路径 |
| `report_format` | string | `"json"` | 报告格式（目前仅支持 `json`） |

---

### 流水线（`pipeline`）

```yaml
pipeline:
  process_every_n_frames: 1
  buffer_size: 1
  max_concurrent: 5
  alert_cooldown: 5
  display: true
  display_scale: 0.1
  camera_interval: 0.1
  sustained_detection_frames: 1
  concurrent_mode: false
  max_queued_frames: 50
  display_input: false
  display_output: true
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `process_every_n_frames` | int | `1` | 每 N 帧触发一次行为分析。`1` = 每帧都分析；增大可降低 API 调用频率 |
| `buffer_size` | int | `1` | 历史帧缓冲区大小。仅追踪模式生效，控制时间窗口范围 |
| `max_concurrent` | int | `5` | 最大并发 API 请求数。`1` = 串行；`≥2` = 启用并发线程池 |
| `alert_cooldown` | int | `5` | 同一行为告警冷却时间（秒）。防止同一行为反复告警 |
| `display` | bool | `true` | 是否显示实时画面。服务器/无头环境设为 `false` |
| `display_scale` | float | `0.1` | 视频窗口缩放比例。`0.5` = 缩小一半 |
| `camera_interval` | float | `0.1` | 摄像头读取间隔（秒）。控制摄像头调用频率 |
| `sustained_detection_frames` | int | `1` | 连续 N 帧检测到目标才触发 API。减少误触发 |
| `concurrent_mode` | bool | `false` | 是否启用外层并发模式。见下方说明 |
| `max_queued_frames` | int | `50` | 并发模式下最大队列帧数。防止内存溢出 |
| `display_input` | bool | `false` | 是否显示原始输入画面（功能 4） |
| `display_output` | bool | `true` | 是否显示标注后的输出画面（功能 4） |

**`concurrent_mode` 两种模式对比：**

```
级联模式 (concurrent_mode: false):
  帧 → YOLO → [等待Qwen结果] → 帧 → YOLO → [等待Qwen结果] → ...

并发模式 (concurrent_mode: true):
  帧 → YOLO → 帧 → YOLO → 帧 → YOLO → ...
                    ↓             ↓             ↓
              提交Qwen任务    提交Qwen任务    提交Qwen任务
                    ↓             ↓             ↓
              异步出队结果    异步出队结果    异步出队结果
```

当 `max_queued_frames` 队列满时，新任务会被丢弃并打印警告日志。

---

### 摄像头日志（`camera_log`）

```yaml
camera_log:
  enabled: true
  retention_hours: 2.0
  log_filename: "camera_behavior_log.json"
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用摄像头行为日志。仅摄像头/RTSP 模式生效 |
| `retention_hours` | float | `2.0` | 日志保留时长（小时）。超过自动删除过期条目 |
| `log_filename` | string | `"camera_behavior_log.json"` | 日志文件名 |

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

### 5. 并发模式

```bash
python main.py --source video --input video.mp4 --concurrent --max-concurrent 5
```

### 6. 视频文件运行示例

```bash
# 水域救助场景
python main.py --source video --input /media/ddc/新加卷/hys/hysnew/agent2/agent2/waterhelping_231125.mp4 --no-display

# 危险行为场景
python main.py --source video --input /media/ddc/新加卷/hys/qmy/agentold/danger_260327.mp4 --no-display

# 溺水场景
python main.py --source video --input /media/ddc/新加卷/hys/hysnew/agent2/agent2/drowning_240112.mp4 --no-display
```

### 7. vLLM 本地模型部署命令

详见 [vllm_serve.md](vllm_serve.md)，包含 4 种模型的启动命令和参数说明。

### 8. 完整命令行参数

```bash
python main.py \
  --source video \
  --input /path/to/video.mp4 \
  --model-mode local \
  --no-display \
  --config config.yaml \
  --output output/ \
  --display-scale 0.5 \
  --max-concurrent 5 \
  --concurrent \
  --verbose
```

| 参数 | 说明 |
|------|------|
| `--source, -s` | 输入源：`video` / `camera` / `rtsp`（必填） |
| `--input, -i` | 视频文件路径 |
| `--camera-id` | USB 摄像头 ID（默认 0） |
| `--rtsp-url` | RTSP 流地址 |
| `--model-mode` | 模型模式：`api` / `local`（优先于 config） |
| `--api-key` | 千问 API Key |
| `--config, -c` | 配置文件路径（默认 config.yaml） |
| `--output, -o` | 输出目录 |
| `--no-display` | 无头模式（服务器环境） |
| `--no-crops` | 不保存人体裁剪图 |
| `--camera-interval` | 摄像头读取间隔（秒） |
| `--display-scale` | 窗口缩放比例 |
| `--max-concurrent` | 最大并发数 |
| `--concurrent` | 启用外层并发模式 |
| `--show-input` | 同时显示原始输入画面 |
| `--verbose, -v` | 详细日志 |

> 命令行参数优先级高于 `config.yaml`。未指定的命令行参数不会覆盖配置文件中的值。

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

## 性能监控

系统内置三类实时速度统计，每 **2 秒**打印一次到终端和日志，并叠加到画面上。

### 终端打印格式

```
[Speed] YOLO:  28.3 frames/s (infer   35.2ms, min=32.1 max=41.5ms) | Stream:  156.20 MB/s | Qwen:   0.42 req/s (avg 2300.0ms, count=15)
```

### 画面叠加格式

当 `display_output: true` 时，画面左上角会叠加类似信息：

```
YOLO: 28.3 frames/s | infer 35.2ms | Stream: 156.2 MB/s | Qwen: 0.42 req/s (avg 2300ms)
```

### 运行结束摘要

```
============================================================
分析完成! 摘要:
  YOLO: 28.3 frames/s (infer avg=35.2ms min=32.1 max=41.5ms, 1360 次)
  Stream: 126.84 MB/s (total 5748.2 MB, avg 900.0 KB/frame)
  Qwen: 0.42 req/s (avg 2300.0ms, 19 次)
============================================================
```

---

### 三类指标详解

#### 1. YOLO FPS（画面处理帧率 + 推理耗时）

**统计原理：**

系统有两层 YOLO 速度统计，含义不同：

| 统计项 | 来源 | 含义 |
|--------|------|------|
| **frames/s** | `FrameRateTracker` | 主线程实际处理帧的速度，包含缓存命中帧。每处理一帧（无论是否真正执行了 YOLO 推理）都计数 |
| **infer ms** | `PersonDetector.get_fps_stats()` | YOLO 单次推理的真实耗时。仅在实际执行推理时记录，跳帧复用缓存时不计入 |

当 `yolo_skip_frames = 0`（每帧推理）时，两者数值接近。
当 `yolo_skip_frames > 0`（隔帧推理）时，`frames/s` 会高于推理 FPS，因为中间帧复用上次结果不耗推理时间。

**滑动窗口机制：**

- `infer ms` 的 `avg` / `min` / `max` 基于最近 **200 次**推理记录的滑动窗口计算
- 窗口满时自动裁剪保留最近一半，防止长时间运行内存无限增长
- `count` 是累计总推理次数（不丢弃）

**如何读：**

```
YOLO:  28.3 frames/s (infer   35.2ms, min=32.1 max=41.5ms)
       ↑                   ↑              ↑
       实际处理帧率         平均推理耗时     波动范围
       (含缓存命中)         (仅推理帧)      (越小越稳定)
```

- `frames/s` 高 → pipeline 吞吐快
- `infer ms` 低 → YOLO 模型快
- `max - min` 差值大 → 推理耗时波动大，可能是 GPU 争抢或输入尺寸变化

---

#### 2. 码流速度（Stream MB/s）

**统计原理：**

`BitrateTracker` 统计每秒从视频源读取的**原始帧数据量**（BGR 三通道原始像素数据）。

计算方式：
```
Stream MB/s = 滑动窗口内总字节数 / 时间跨度 / 1048576
```

每读取一帧，记录 `(timestamp, frame.nbytes)`，使用 10 秒滑动窗口平滑统计。

**常见参考值：**

| 分辨率 | 每帧大小 | 30fps 时的码流 |
|--------|---------|---------------|
| 320×240 | 225 KB | 6.6 MB/s |
| 640×480 | 900 KB | 26.4 MB/s |
| 1280×720 | 2.7 MB | 79.1 MB/s |
| 1920×1080 | 6.0 MB | 175.8 MB/s |

> 注意：这是**解码后的原始像素数据量**，不等于视频文件的编码码率（后者通常小很多）。

**如何读：**

```
Stream: 156.20 MB/s
        ↑
        原始帧数据吞吐量
```

- 码流 ≈ 分辨率 × 3B × 实际读取帧率 → 正常
- 码流异常低 → 视频源可能丢帧或读取卡顿
- 码流可用于判断 pipeline 瓶颈：如果 YOLO FPS 低但码流正常，瓶颈在 YOLO 推理

**最终摘要额外字段：**

| 字段 | 含义 |
|------|------|
| `total MB` | 运行期间累计读取的总数据量 |
| `avg KB/frame` | 平均每帧数据大小（反映分辨率） |

---

#### 3. Qwen API 吞吐与延迟

**统计原理：**

`QwenFpsTracker` 统计两个维度：

| 维度 | 指标 | 计算方式 |
|------|------|---------|
| **吞吐** | `req/s` | 滑动窗口（10 秒）内每秒完成的 API 请求数。并发模式下反映多线程实际效果 |
| **延迟** | `avg / min / max ms` | 最近 50 次 API 请求的耗时统计 |

**滑动窗口机制：**

- 吞吐统计：基于 `completion_timestamps` 的 10 秒滑动窗口
- 延迟统计：基于最近 50 次 `_inference_times` 的滑动窗口
- `_inference_times` 列表上限 200 条，超出时裁剪保留最近一半

**级联模式 vs 并发模式的区别：**

```
级联模式:
  Qwen:   0.15 req/s (avg 6500.0ms, count=5)
          ↑                ↑
          吞吐 = 1/延迟     每次等完再做下一次

并发模式 (max_concurrent=5):
  Qwen:   0.68 req/s (avg 3200.0ms, count=25)
          ↑                ↑
          吞吐 > 1/延迟     多请求并发，实际完成更快
```

**如何读：**

```
Qwen:   0.42 req/s (avg 2300.0ms, count=15)
        ↑              ↑              ↑
        API 吞吐量      平均延迟        累计完成次数
        (越高越好)      (越低越好)      (总量)
```

- `req/s` 高 → API 调用效率高（并发模式会显著提升）
- `avg ms` 低 → 模型响应快
- `count` 持续增长 → 系统正常工作
- `req/s = 0` 且 `count = 0` → 还没有 API 请求完成（可能还在预热或无检测目标）

---

### 数据流示意

```
视频源帧 ──> BitrateTracker.record(frame)     ← 每帧都记录
    │
    v
YOLO detect ──> detector._inference_times     ← 仅实际推理时记录
    │              │
    v              v
FrameRateTracker   get_fps_stats()
    │                   │
    v                   v
 [每2秒打印] ──── _print_fps_stats() ──── 终端 + 画面叠加
                        │
                        ├── YOLO: frames/s (infer ms, min/max)
                        ├── Stream: MB/s
                        └── Qwen: req/s (avg ms, count)
```

---

## 项目结构

```
actionv5/
├── main.py                        # 主入口
├── config.yaml                    # 全局配置
├── prompts/                       # 提示词模板
│   ├── detailed_prompt.txt        #   详细版提示词
│   └── brief_prompt.txt           #   简练版提示词
├── core/
│   ├── __init__.py
│   ├── detector.py                #   YOLO 人体检测 + ByteTrack/BotSORT 跟踪 + 隔帧推理
│   ├── pipeline.py                #   推理流水线（级联/并发模式、FPS/码流统计、告警、日志）
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
├── analysis_report.json           # 分析报告（帧数、行为统计、运行时长、速度统计）
├── alerts.json                    # 告警记录（仅存在告警时生成）
├── camera_behavior_log.json       # 摄像头行为日志（自动清理过期条目，仅摄像头模式）
├── annotated/                     # 标注后的帧图片（带检测框和行为标签）
└── crops/                         # 人体裁剪图
```

`analysis_report.json` 示例：
```json
{
  "source": "video",
  "duration_seconds": 45.32,
  "total_frames": 1360,
  "processed_frames": 1360,
  "total_detections": 892,
  "behavior_counts": {
    "0": 3,
    "1": 456,
    "3": 433
  },
  "alert_count": 3
}
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
- 启用隔帧推理：`yolo_skip_frames: 2`（每 3 帧推理 1 次）

### Q: 并发模式下结果延迟
- `max_queued_frames` 队列满时会丢弃新任务，适当增大
- 检查 Qwen API 响应速度，本地模型可调大 `max_concurrent`
