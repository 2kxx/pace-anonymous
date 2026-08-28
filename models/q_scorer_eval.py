"""
Q-Scorer Wrapper
封装基于 LLaVA 架构微调的图像质量评估模型
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import List, Dict, Union, Any
from dataclasses import dataclass

# ======= 导入你自己的 Q-Scorer 依赖 =======
from models.qscorer.model.builder import load_pretrained_model
from models.qscorer.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from models.qscorer.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


# ======= 你的核心 Scorer 模型 =======
class Scorer(nn.Module):
    def __init__(self, model_path="", model_base="", preprocessor_path="", device="cuda:0"):
        super().__init__()

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            model_base,
            "qscorer_lora",
            preprocessor_path=preprocessor_path,
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

        self.preferential_ids_ = [tokenizer.convert_tokens_to_ids(['<score5>', '<score4>', '<score3>', '<score2>', '<score1>'])]

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    def expand2square(self, pil_img, background_color):
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
        image = [self.expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean)) for img in image]
        with torch.inference_mode():
            image_tensors = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(self.model.device)
            embedding = None
            current_input = self.input_ids.repeat(image_tensors.shape[0], 1)

            for i in range(2):
                output = self.model(
                    input_ids=current_input,
                    images=image_tensors,
                    output_hidden_states=True
                )
                logits = output["logits"][:, -1]
                probs = torch.softmax(logits, dim=-1)
                if i == 1:
                    embedding = output["hidden_states"][:, -1, :]

                vocab_ids = torch.argmax(probs, dim=-1)
                current_input = torch.cat([current_input, vocab_ids.unsqueeze(1)], dim=1)

            scores = self.model.deepmlp(embedding)
            return scores


# ======= 适配 PlanningAgent 的 Wrapper =======
class QScorerWrapper:
    """
    Q-Scorer 包装器
    负责处理 Agent 传来的 Numpy 图像，并格式化输出
    """
    def __init__(self, model_path: str, model_base: str = None, preprocessor_path: str = "./models/qscorer/preprocessor", device: str = "cuda:0", **kwargs):
        print(f"  -> 正在加载 Q-Scorer VLM 模型...")
        self.model = Scorer(
            model_path=model_path,
            model_base=model_base,
            preprocessor_path=preprocessor_path,
            device=device
        )
        self.model.eval()

    def score(self, image: Union[str, np.ndarray, Image.Image]) -> float:
        """标准评估接口"""
        # 1. 类型转换：Numpy -> PIL.Image (满足 Scorer 的要求)
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype('uint8')).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Q-Scorer 收到不支持的图像类型: {type(image)}")

        # 2. 执行模型前向传播
        try:
            # 返回的是 Tensor, 取出实际的 float 数值
            raw_score_tensor = self.model([pil_img])

            # 处理张量提取标量：如果 shape 是 [1, 1] 等情况
            if raw_score_tensor.dim() > 0:
                score_val = float(raw_score_tensor.flatten()[0].item())
            else:
                score_val = float(raw_score_tensor.item())

            # 限制分数在 1.0 - 5.0 的合法区间内
            final_score = max(1.0, min(5.0, score_val))
            return final_score
        except Exception as e:
            print(f"  │ [Q-Scorer Error] 推理失败: {e}")
            return 3.0  # 失败兜底分


def create_q_scorer(model_path: str, model_base: str, **kwargs) -> QScorerWrapper:
    """系统初始化工厂函数"""
    return QScorerWrapper(model_path=model_path, model_base=model_base, **kwargs)