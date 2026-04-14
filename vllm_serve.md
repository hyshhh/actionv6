# vLLM 模型服务部署配置

## 启动命令

以下为不同模型的 vLLM 启动命令，按需选用。

> `--api-key` 和 `--served-model-name` 需与 `config.yaml` 中保持一致。

### 1. Qwen3-VL-4B-Instruct（原始精度）

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/qmy/qwen \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --max-model-len 1024 \
  --port 7890 \
  --gpu-memory-utilization 0.25
```

### 2. Qwen3-VL-4B-AWQ-4bit（4bit 量化）

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew/Qwen3-VL-4B-Instruct-AWQ-4bit \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 1024 \
  --port 7890 \
  --gpu-memory-utilization 0.25
```

### 3. Qwen3.5-2B（轻量版）

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 1024 \
  --port 7890 \
  --gpu-memory-utilization 0.25
```

### 4. Qwen3.5-2B-AWQ（轻量量化，长上下文）

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 10240 \
  --port 7890 \
  --gpu-memory-utilization 0.15 \
  --max-num-seqs 10
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `CUDA_VISIBLE_DEVICES` | 指定使用的 GPU 编号 |
| `--api-key` | API 密钥，需与 config.yaml 中 `qwen.api_key` 一致 |
| `--served-model-name` | 模型名称标识，需与 config.yaml 中 `qwen.local_model_name` 一致 |
| `--max-model-len` | 最大上下文长度（token） |
| `--port` | 服务端口号 |
| `--gpu-memory-utilization` | GPU 显存占用比例 |
| `--max-num-seqs` | 最大并发序列数 |

## config.yaml 对应配置

使用本地模型时，`config.yaml` 中需配置：

```yaml
qwen:
  model_mode: local
  api_key: "abc123"
  local_model_name: "Qwen/Qwen3-VL-4B-Instruct"  # 与 --served-model-name 一致
  local_base_url: "http://localhost:7890/v1"       # 与 --port 一致
```
