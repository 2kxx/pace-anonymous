"""
Qwen2.5VL 多模态大模型本地加载与推理模块
适配 PlanningAgent 的感知生长框架
"""

from typing import List, Dict, Union, Optional, Tuple, Any
from dataclasses import dataclass
import json
import re

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

@dataclass
class VQAResponse:
    """VQA响应结果"""
    text: str
    confidence: float = 1.0
    reasoning: str = ""
    raw_output: Optional[Dict] = None


class Qwen25VLModel:
    """Qwen2.5-VL 7B 本地模型封装类"""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        **kwargs
    ):
        self.model_path = model_path
        self.device = device
        self.dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        self.torch_dtype = self.dtype_map.get(torch_dtype, torch.bfloat16)

        self.model = None
        self.processor = None
        self._initialized = False

    def load_model(self):
        if self._initialized:
            return

        print(f"正在加载Qwen2.5-VL模型: {self.model_path}")
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True
            )

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            self.model.eval()
            self._initialized = True
            print("Qwen2.5-VL模型加载完成！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def chat(
        self,
        text: str,
        image: Union[str, Image.Image, List[Union[str, Image.Image]], None] = None,
        **kwargs
    ) -> str:
        """
        统一的对话接口，适配 PlanningAgent 的调用方式。
        支持：纯文本、单图+文本、多图+文本
        """
        # 归一化 image 参数为列表，如果是 None 则为空列表
        images = []
        if image is not None:
            if isinstance(image, list):
                images = image
            else:
                images = [image]

        # 调用底层推理
        response = self.inference(images=images, prompt=text, **kwargs)
        return response.text

    def inference(
        self,
        images: List[Union[str, Image.Image]],
        prompt: str,
        system_prompt: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs
    ) -> VQAResponse:
        if not self._initialized:
            self.load_model()

        # 准备消息结构
        messages = self._prepare_messages(images, prompt, system_prompt)

        # 应用聊天模板
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 处理视觉信息（如果有）
        image_inputs, video_inputs = process_vision_info(messages)

        # 构建输入参数
        process_kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt"
        }

        # 只有存在视觉输入时才添加 images/videos 参数
        # process_vision_info 在无图时通常返回 None
        if image_inputs is not None:
            process_kwargs["images"] = image_inputs
        if video_inputs is not None:
            process_kwargs["videos"] = video_inputs

        inputs = self.processor(**process_kwargs)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True if temperature > 0 else False
            )

        # 解码输出
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = self.processor.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return VQAResponse(
            text=response_text,
            confidence=1.0, # 这里暂时硬编码，Qwen output scores需要额外处理才能转confidence
            raw_output={"generated_ids": generated_ids.cpu().tolist()}
        )

    def predict_probs(
            self,
            text: str,
            image: Union[str, Image.Image, List[Union[str, Image.Image]], None] = None,
            candidates: List[str] = ["A", "B", "C"],
            **kwargs
    ) -> Dict[str, float]:
        """
        获取指定候选 Token (如 A, B, C) 的 Softmax 概率分布
        """
        if not self._initialized:
            self.load_model()

        # 1. 处理多模态输入
        images = image if isinstance(image, list) else ([image] if image is not None else [])
        messages = self._prepare_messages(images, text)
        input_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(text=[input_text], images=image_inputs, videos=video_inputs, padding=True,
                                return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 2. 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取最后一个位置的 logits
            last_logits = outputs.logits[:, -1, :]

        # 3. 提取候选 ID 并计算 Softmax
        # 这里的技巧是取每个字母的首个 token id
        candidate_ids = [self.processor.tokenizer.encode(c, add_special_tokens=False)[-1] for c in candidates]

        # 只取 A, B, C 对应的 Logits
        relevant_logits = last_logits[0, candidate_ids]
        # 使用 Softmax 归一化，得到概率分布
        probs = torch.softmax(relevant_logits.float(), dim=-1).cpu().numpy()

        return {cand: float(prob) for cand, prob in zip(candidates, probs)}

    def _prepare_messages(
        self,
        images: List[Union[str, Image.Image]],
        prompt: str,
        system_prompt: str = None
    ) -> List[Dict]:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = []

        # 处理图片（如果有）
        if images:
            for img in images:
                # 简单校验，防止 None 混入
                if img is None: continue

                if isinstance(img, str):
                    # 也可以添加 base64 或 http 校验
                    content.append({"type": "image", "image": img})
                elif isinstance(img, Image.Image):
                    content.append({"type": "image", "image": img})
                elif isinstance(img, np.ndarray):
                    # 增加对 numpy array 的支持 (Agent 经常传 numpy)
                    content.append({"type": "image", "image": Image.fromarray(img)})

        # 添加文本 prompt
        content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": content})

        return messages


# ---------------------------------------------------------------------------
# 统一 VQA 模型工厂
# ---------------------------------------------------------------------------
# 通过 ``backend`` 参数在不同的多模态大模型之间切换：
#   - "qwen"        -> Qwen2.5-VL（默认）
#   - "mplug_owl2"  -> mPLUG-Owl2-LLaMA2-7B
# 任意 backend 返回的对象都实现了 ``chat() / inference() / predict_probs()`` 三件套，
# 因此上层 PlanningAgent / VisualizerAgent / CriticAgent / GeneratorAgent 无需感知差异。
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_PATHS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "mplug_owl2": "MAGAer13/mplug-owl2-llama2-7b",
    "mplug-owl2": "MAGAer13/mplug-owl2-llama2-7b",
}


def create_vqa_model(
    model_path: Optional[str] = None,
    backend: str = "qwen",
    **kwargs,
):
    """创建 VQA 模型实例。

    Args:
        model_path: 模型权重路径或 HF repo id。如果为 None，会根据 ``backend`` 选择默认。
        backend: 后端名称，目前支持 ``"qwen"`` 与 ``"mplug_owl2"``。
        **kwargs: 透传给具体 backend 的初始化参数（device / torch_dtype 等）。
    """
    backend_key = (backend or "qwen").lower().strip()
    if model_path is None:
        model_path = _DEFAULT_MODEL_PATHS.get(backend_key, _DEFAULT_MODEL_PATHS["qwen"])

    if backend_key in {"qwen", "qwen2.5-vl"}:
        return Qwen25VLModel(model_path=model_path, **kwargs)

    if backend_key in {"mplug_owl2", "mplug-owl2"}:
        # 延迟导入：避免在不使用 mPLUG-Owl2 时强行触发其依赖（peft、icecream 等）
        from models.mplug_owl2_model import MPLUGOwl2Model

        # mPLUG-Owl2 推荐 fp16；如果调用方传了 bfloat16/float32，这里给出友好提示
        if kwargs.get("torch_dtype") and kwargs["torch_dtype"] != "float16":
            print(
                f"[create_vqa_model] mPLUG-Owl2 推荐 fp16，当前传入 "
                f"torch_dtype={kwargs['torch_dtype']!r}，将由 builder 内部强制 fp16。"
            )
        return MPLUGOwl2Model(model_path=model_path, **kwargs)

    raise ValueError(
        f"未知的 VQA backend: {backend!r}，可选: {sorted(set(_DEFAULT_MODEL_PATHS.keys()))}"
    )