"""行为分类器 — 千问多模态模型（云端 API / 本地 vLLM）"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from models.schemas import BehaviorResult, Severity
from utils.logger import get_logger

logger = get_logger()

# 提示词模板文件目录（相对于项目根目录）
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class BehaviorClassifier:
    """
    调用多模态大模型进行行为识别。
    支持两种模式：
    - api:   调用云端 API（阿里云百炼千问）
    - local: 调用本地 vLLM 部署的模型（OpenAI 兼容接口）

    工作流程：
    1. 将人体区域关键帧（base64）发送给模型
    2. 模型分析连续帧中的动作变化
    3. 返回结构化的行为标签和描述
    """

    # 行为严重度映射
    _SEVERITY_MAP = {
        "critical": Severity.CRITICAL,
        "warning": Severity.WARNING,
        "normal": Severity.NORMAL,
    }

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-vl-max",
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout: int = 30,
        behavior_classes: list[dict] | None = None,
        model_mode: str = "api",
        prompt_mode: str = "detailed",
    ):
        """
        Args:
            api_key: API Key（云端模式为千问 Key，本地模式为 vLLM --api-key）
            api_url: API 地址（OpenAI 兼容格式）
            model: 模型名称 / served-model-name
            max_tokens: 最大生成 token 数
            temperature: 生成温度
            timeout: 请求超时（秒）
            behavior_classes: 行为类别配置列表
            model_mode: "api"（云端）或 "local"（本地 vLLM）
            prompt_mode: "detailed"（详细提示词）或 "brief"（简练提示词）
        """
        self.model_mode = model_mode
        self.prompt_mode = prompt_mode
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "")
        self.api_url = api_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        if model_mode == "api" and not self.api_key:
            logger.warning("云端 API 模式：API Key 未设置！请在 config.yaml 或环境变量 QWEN_API_KEY 中配置")
        elif model_mode == "local":
            logger.info(f"本地模型模式: endpoint={api_url}, model={model}")
            # 本地模式 api_key 可以为空（vLLM 未设置 --api-key 时）
            if not self.api_key:
                self.api_key = "EMPTY"  # vLLM 无鉴权时的占位符

        # 初始化 OpenAI 兼容客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
            timeout=self.timeout,
        )

        # 加载行为类别定义
        self.behavior_classes = behavior_classes or self._default_classes()
        self._build_prompt()

        logger.info(
            f"行为分类器初始化完成: mode={model_mode}, model={model}, "
            f"categories={len(self.behavior_classes)}"
        )

    @staticmethod
    def _default_classes() -> list[dict]:
        """默认行为类别"""
        return [
            {
                "id": "0",
                "label_cn": "溺水",
                "label_en": "drowning",
                "severity": "critical",
                "description": "四肢无规律挣扎，有溺水风险。",
            },
            {
                "id": "1",
                "label_cn": "游泳",
                "label_en": "swimming",
                "severity": "normal",
                "description": "人员在水中正常游泳。",
            },
            {
                "id": "2",
                "label_cn": "攀爬栏杆",
                "label_en": "climbing",
                "severity": "warning",
                "description": "人员攀爬或翻越栏杆。",
            },
            {
                "id": "3",
                "label_cn": "正常行走",
                "label_en": "normal_walking",
                "severity": "normal",
                "description": "岸上人员正常行走或站立。",
            },
            {
                "id": "4",
                "label_cn": "正在救援",
                "label_en": "waterhelping",
                "severity": "normal",
                "description": "水中人员抱住红色救生圈。",
            },
            {
                "id": "5",
                "label_cn": "在船上",
                "label_en": "abord",
                "severity": "normal",
                "description": "人员在船上或在开船。",
            },
        ]

    def _build_prompt(self):
        """从外部模板文件构建系统提示词"""
        categories_text = ""
        for cls in self.behavior_classes:
            categories_text += (
                f"- **{cls['id']}** {cls['label_en']}): "
                f"{cls['description'].strip()} "
                f"[severity: {cls['severity']}]\n"
            )

        valid_ids = [cls["id"] for cls in self.behavior_classes]

        # 根据 prompt_mode 选择模板文件
        template_filename = f"{self.prompt_mode}_prompt.txt"
        template_path = _PROMPTS_DIR / template_filename

        if not template_path.exists():
            logger.warning(f"提示词模板文件不存在: {template_path}, 回退到详细版")
            template_path = _PROMPTS_DIR / "detailed_prompt.txt"

        if not template_path.exists():
            # 兜底：模板文件全部丢失时使用内置默认提示词
            logger.error("所有提示词模板文件缺失，使用内置默认提示词")
            self.system_prompt = (
                "你是一个专业的救生行为分析AI。你的任务是识别画面中人物的行为。\n\n"
                "## 可识别的行为类别\n"
                f"{categories_text}\n"
                "## 输出要求\n"
                "请严格按 JSON 格式输出 behavior_id、behavior_label、description、severity、confidence。\n"
                f"behavior_id 必须是以下之一: {valid_ids}\n"
                "如果无法确定行为，返回 unknown。"
            )
            return

        template = template_path.read_text(encoding="utf-8")
        self.system_prompt = template.format(
            categories_text=categories_text,
            valid_ids=valid_ids,
        )
        logger.info(f"已加载提示词模板: {template_filename}")

    def classify(
        self,
        keyframes_b64: list[str],
        context: str = "",
    ) -> BehaviorResult:
        """
        分析一组关键帧，返回行为识别结果。

        Args:
            keyframes_b64: base64 编码的关键帧列表（至少1帧）
            context: 额外上下文信息（如"泳池区域"）

        Returns:
            BehaviorResult 行为识别结果
        """
        if not keyframes_b64:
            return BehaviorResult(
                behavior_id="unknown",
                behavior_label="未知",
                description="未提供关键帧",
                severity=Severity.NORMAL,
            )

        # 构建用户消息（多模态）
        user_content = []

        # 添加文字说明
        frame_count = len(keyframes_b64)
        instruction = (
            f"以下是 {frame_count} 张连续关键帧（同一人物的连续动作），"
            "请分析该人物的行为。"
        )
        if context:
            instruction += f"\n场景上下文: {context}"

        user_content.append({"type": "text", "text": instruction})

        # 添加图像
        for i, b64 in enumerate(keyframes_b64):
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            })

        # 添加收尾指令
        user_content.append({
            "type": "text",
            "text": f"请根据以上 {frame_count} 张关键帧分析行为，严格按 JSON 格式输出。",
        })

        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            elapsed = time.time() - start_time
            mode_label = "本地" if self.model_mode == "local" else "云端"
            logger.debug(f"{mode_label}模型调用完成, 耗时 {elapsed:.2f}s")

            # 解析响应
            content = response.choices[0].message.content or ""
            return self._parse_response(content)

        except Exception as e:
            mode_label = "本地" if self.model_mode == "local" else "云端"
            logger.error(f"{mode_label}模型调用失败: {e}")
            return BehaviorResult(
                behavior_id="unknown",
                behavior_label="未知",
                description=f"API 调用失败: {str(e)}",
                severity=Severity.NORMAL,
            )

    def _parse_response(self, content: str) -> BehaviorResult:
        """
        解析模型的 JSON 响应。

        支持：
        - 纯 JSON 输出
        - JSON 包裹在 ```json ... ``` 代码块中
        - JSON 混杂在其他文本中（尝试提取）
        """
        content = content.strip()

        # 尝试从代码块中提取 JSON
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # 尝试直接解析或找 { } 包裹的内容
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"无法解析模型输出为 JSON: {content[:200]}")
            return BehaviorResult(
                behavior_id="unknown",
                behavior_label="未知",
                description=f"模型输出解析失败: {content[:200]}",
                severity=Severity.NORMAL,
            )

        # 提取字段并验证
        behavior_id = data.get("behavior_id", "unknown")
        behavior_label = data.get("behavior_label", "未知")
        description = data.get("description", "")
        severity_str = data.get("severity", "normal")
        confidence = float(data.get("confidence", 0.0))

        # 调试日志：显示模型返回的原始值
        logger.debug(f"模型返回: behavior_id='{behavior_id}', behavior_label='{behavior_label}', severity='{severity_str}', confidence={confidence}")

        # 验证 behavior_id 是否在允许列表中
        valid_ids = [cls["id"] for cls in self.behavior_classes]
        logger.debug(f"有效行为ID列表: {valid_ids}")
        if behavior_id not in valid_ids:
            # 尝试模糊匹配
            matched = False
            for cls in self.behavior_classes:
                if cls["id"] in behavior_id or cls["label_cn"] in behavior_label:
                    behavior_id = cls["id"]
                    behavior_label = cls["label_cn"]
                    severity_str = cls["severity"]
                    matched = True
                    break
            if not matched:
                logger.warning(f"未知行为 ID: {behavior_id}, 回退为 unknown")
                behavior_id = "unknown"

        # 映射严重度
        severity = self._SEVERITY_MAP.get(severity_str, Severity.NORMAL)

        return BehaviorResult(
            behavior_id=behavior_id,
            behavior_label=behavior_label,
            description=description,
            severity=severity,
            confidence=confidence,
        )

    def classify_single(self, crop_b64: str) -> BehaviorResult:
        """快捷方法：分析单帧裁剪图"""
        return self.classify([crop_b64])
