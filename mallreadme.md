# 智能救生行为识别 Agent (ActionV5)

基于 YOLOv8 + Qwen 多模态大模型的水域安全行为识别系统。

## 功能特性

- YOLOv8/v11 人体检测 + ByteTrack 目标跟踪
- SAHI 切片推理（提升小目标检出率）
- Qwen 多模态大模型行为分类（云端 API / 本地 vLLM）
- 实时视频流处理（摄像头 / RTSP / 视频文件）
- 行为标签持久化跟踪（VLM 低频推理场景）

## 运行示例

### 视频文件分析

```bash
# 正常游泳场景
python main.py --source video --input /media/ddc/新加卷/hys/qmy/agentold/agent2v1/1.mp4 --no-display

# 救援场景
python main.py --source video --input /media/ddc/新加卷/hys/hysnew/agent2/agent2/waterhelping_231125.mp4 --no-display

# 危险场景
python main.py --source video --input /media/ddc/新加卷/hys/qmy/agentold/danger_260327.mp4 --no-display

# 溺水场景
python main.py --source video --input /media/ddc/新加卷/hys/hysnew/agent2/agent2/drowning_240112.mp4 --no-display
```

### 其他用法

```bash
# USB 摄像头
python main.py --source camera --camera-id 0

# RTSP 流
python main.py --source rtsp --rtsp-url rtsp://192.168.1.100:554/stream

# 本地 vLLM 模型
python main.py --source video --input video.mp4 --model-mode local

# 保存标注视频
python main.py --source video --input video.mp4 --save-video

# 启用 SAHI（小目标增强）
# 修改 config.yaml: sahi_enabled: true
```

## 配置说明

主要配置项在 `config.yaml`：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| model_mode | 模型模式 (api/local) | local |
| detector.confidence | 检测置信度 | 0.25 |
| detector.sahi_enabled | SAHI 切片推理 | false |
| detector.sahi_batch_enabled | SAHI 批量推理 | true |
| pipeline.process_every_n_frames | VLM 调用频率 | 1 |
| pipeline.concurrent_mode | 并发模式 | false |
