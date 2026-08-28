"""
DeQA-Score Evaluation Module
基于 mplug_owl2 架构的图像质量评估模型包装器
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import List, Union, Any

# 导入 DeQA 专有的构建工具（假设这些路径在你的 PYTHONPATH 中）
from models.deqa.model.builder import load_pretrained_model
from models.deqa.constants import IMAGE_TOKEN_INDEX
from models.deqa.mm_utils import tokenizer_image_token


class Scorer(nn.Module):
    """
    DeQA-Score 核心模型类
    计算 "excellent", "good", "fair", "poor", "bad" 五个 Token 的概率加权分
    """
    def __init__(self, pretrained="zhiyuanyou/DeQA-Score-Mix3", device="cuda:0"):
        super().__init__()
        print(f"  -> 正在加载 DeQA-Score 模型 (mplug_owl2): {pretrained}")

        # 加载预训练模型
        tokenizer, model, image_processor, _ = load_pretrained_model(
            pretrained,
            None,
            "mplug_owl2",
            device=device
        )
        missing_attributes = {
            '_use_flash_attention_2': False,
            '_use_sdpa': False,
            '_use_bettertransformer': False,
        }

        # 定义一个递归注入函数，确保内部的 Llama 模型也能被注入
        def inject_attrs(target_obj):
            for attr, val in missing_attributes.items():
                if not hasattr(target_obj, attr):
                    setattr(target_obj, attr, val)

        # 1. 注入顶层模型
        inject_attrs(model)

        # 2. 注入内部的语言模型（mPLUG 结构中通常在 model.model）
        if hasattr(model, 'model'):
            inject_attrs(model.model)

        # 3. 针对 Llama 结构可能存在的更深层引用
        if hasattr(model, 'get_model'):
            inject_attrs(model.get_model())

        prompt = "USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is"

        # 提取质量评价相关的 Token ID
        # 对应权重: excellent=5, good=4, fair=3, poor=2, bad=1
        self.preferential_ids_ = [id_[1] for id_ in tokenizer(["excellent", "good", "fair", "poor", "bad"])["input_ids"]]
        self.weight_tensor = torch.Tensor([5., 4., 3., 2., 1.]).half().to(model.device)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        # 预计算输入 ID
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    def expand2square(self, pil_img, background_color):
        """将图像补全为正方形，防止变形"""
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def forward(self, image: List[Image.Image]):
        # 1. 图像预处理：补齐正方形
        bg_color = tuple(int(x * 255) for x in self.image_processor.image_mean)
        image = [self.expand2square(img, bg_color) for img in image]

        with torch.inference_mode():
            # 2. 提取特征
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(self.model.device)

            # 3. 获取最后一个 Token 的 Logits 并切片到指定的 5 个质量词上
            output_logits = self.model(
                input_ids=self.input_ids.repeat(image_tensor.shape[0], 1),
                images=image_tensor
            )["logits"][:, -1, self.preferential_ids_]

            # 4. Softmax 归一化并进行加权求和得出 1-5 分
            return torch.softmax(output_logits, -1) @ self.weight_tensor


class DeQAScoreWrapper:
    """
    DeQA-Score 包装器
    适配 PlanningAgent，提供统一的 .score(image) 接口
    """
    def __init__(self, model_path="zhiyuanyou/DeQA-Score-Mix3", device="cuda:0", **kwargs):
        self.device = device
        self.model = Scorer(pretrained=model_path, device=device)
        self.model.eval()

    def score(self, image: Union[str, np.ndarray, Image.Image]) -> float:
        """核心评估接口"""
        # 1. 统一转换为 PIL Image
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype('uint8')).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # 2. 模型推理 (注意：Scorer.forward 接收的是 List)
        try:
            score_tensor = self.model([pil_img])
            # 提取标量数值
            score_val = float(score_tensor.flatten()[0].item())
            # 边界保护
            return max(1.0, min(5.0, score_val))
        except Exception as e:
            print(f"  │ [DeQA Error] 推理失败: {e}")
            return 3.0  # 失败兜底分


def create_deqa_scorer(model_path="zhiyuanyou/DeQA-Score-Mix3", device="cuda:0", **kwargs):
    """工厂函数"""
    return DeQAScoreWrapper(model_path=model_path, device=device, **kwargs)