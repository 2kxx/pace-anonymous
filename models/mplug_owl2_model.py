"""
mPLUG-Owl2 多模态大模型本地加载与推理模块。

对外接口与 ``models.vqa_model.Qwen25VLModel`` 完全对齐
（``chat()`` / ``inference()`` / ``predict_probs()``），
因此 PlanningAgent / VisualizerAgent / CriticAgent / GeneratorAgent
等所有上层组件无需任何改动即可切换到 mPLUG-Owl2 作为推理大脑。

参考：
    - DeQA-Score 已经使用相同的 ``models.deqa.model.MPLUGOwl2LlamaForCausalLM``
      作为底层骨架（仅用于打分），这里复用同一套权重加载与图像 token 注入逻辑，
      将它升级为通用的“感知大脑”，支持多图、纯文本与 Logits 概率分布查询。
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from models.vqa_model import VQAResponse

# 复用 DeQA 已经搬运到仓库内的 mPLUG-Owl2 推理组件
from models.deqa.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from models.deqa.conversation import conv_templates
from models.deqa.mm_utils import process_images, tokenizer_image_token
from models.deqa.model.builder import load_pretrained_model


class MPLUGOwl2Model:
    """mPLUG-Owl2 通用 VLM 封装类。

    Args:
        model_path: HuggingFace repo id 或本地权重目录。默认 ``MAGAer13/mplug-owl2-llama2-7b``。
        device: 设备字符串。
        torch_dtype: 计算精度（``"float16"`` / ``"bfloat16"`` / ``"float32"``）。
            注：mPLUG-Owl2 原仓库训练精度是 fp16，不建议使用 bf16/fp32。
    """

    def __init__(
        self,
        model_path: str = "MAGAer13/mplug-owl2-llama2-7b",
        device: str = "cuda",
        torch_dtype: str = "float16",
        conv_template: str = "mplug_owl2",
        **kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.device = device
        # mPLUG-Owl2 强烈依赖 fp16，这里只是保留接口，实际加载时由 builder.py 控制
        self.torch_dtype_str = torch_dtype
        self.conv_template = conv_template

        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self._initialized = False

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        if self._initialized:
            return

        print(f"正在加载 mPLUG-Owl2 模型: {self.model_path}")
        try:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name="mplug_owl2",
                device=self.device,
            )

            # 与 DeQA / Q-Scorer 包装器保持一致：补齐 transformers 新版本的兼容属性
            missing_attrs = {
                "_use_flash_attention_2": False,
                "_use_sdpa": False,
                "_use_bettertransformer": False,
            }

            def _inject(target: Any) -> None:
                for k, v in missing_attrs.items():
                    if not hasattr(target, k):
                        setattr(target, k, v)

            _inject(model)
            if hasattr(model, "model"):
                _inject(model.model)
            if hasattr(model, "get_model"):
                _inject(model.get_model())

            model.eval()

            # 应用 past_key_values 防御补丁（详见方法文档）
            self._apply_pkv_compat_patch()

            self.tokenizer = tokenizer
            self.model = model
            self.image_processor = image_processor
            self.context_len = context_len
            self._initialized = True
            print("mPLUG-Owl2 模型加载完成！")
        except Exception as exc:  # noqa: BLE001
            print(f"mPLUG-Owl2 模型加载失败: {exc}")
            raise

    # ------------------------------------------------------------------
    # 兼容性补丁
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_pkv_compat_patch() -> None:
        """对 DeQA 自带的 ``model_forward`` 打一次进程级 monkey patch。

        DeQA 仓库的 ``models/deqa/model/modeling_llama2.py::model_forward``
        在 ``past_key_values is not None`` 分支里硬性访问
        ``past_key_values[0][0].shape[2]``。在某些 transformers / accelerate
        版本组合下，调用方就算显式传 ``past_key_values=None``，到这一层时也
        会被中间层（DynamicCache 占位 / 模型自有的 cache 初始化）替换成
        ``(None,)*num_layers`` 之类的"空 cache"，触发
        ``TypeError: 'NoneType' object is not subscriptable``。

        这里把 ``model_forward`` 包一层，把 ``(None,)*N`` / 空 ``DynamicCache``
        统一规范化成 ``None``，让原函数的 ``if`` 分支正常短路。
        - 不改动 vendored 源码；
        - 对正常 legacy KV cache 完全透明；
        - 全进程只打一次（``_pace_pkv_patched`` flag）。
        """
        from models.deqa.model import modeling_llama2 as _ml
        import transformers.models.llama.modeling_llama as _hf_llama

        if getattr(_ml, "_pace_pkv_patched", False):
            return

        _orig = _ml.model_forward

        def _normalize_pkv(pkv: Any) -> Any:
            if pkv is None:
                return None
            # DynamicCache -> legacy tuple-of-tuples
            if hasattr(pkv, "to_legacy_cache"):
                try:
                    pkv = pkv.to_legacy_cache()
                except Exception:  # noqa: BLE001
                    return None
            if pkv is None:
                return None
            try:
                if len(pkv) == 0:
                    return None
                first = pkv[0]
                if first is None:
                    return None
                if hasattr(first, "__len__") and hasattr(first, "__getitem__"):
                    if len(first) == 0 or first[0] is None:
                        return None
            except Exception:  # noqa: BLE001
                return None
            return pkv

        def _safe_model_forward(self, *args, **kwargs):
            if "past_key_values" in kwargs:
                kwargs["past_key_values"] = _normalize_pkv(kwargs["past_key_values"])
            return _orig(self, *args, **kwargs)

        # 同步替换两个引用：vendored 模块内的、以及 transformers.LlamaModel 上的
        # （后者是 replace_llama_modality_adaptive() 在 import 时写入的）
        _ml.model_forward = _safe_model_forward
        _hf_llama.LlamaModel.forward = _safe_model_forward
        _ml._pace_pkv_patched = True

    # ------------------------------------------------------------------
    # 工具：图像归一化 + tensor 化
    # ------------------------------------------------------------------
    @staticmethod
    def _to_pil(image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image.astype("uint8")).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise ValueError(f"Unsupported image type for mPLUG-Owl2: {type(image)}")

    def _build_image_tensor(
        self, images: List[Union[str, np.ndarray, Image.Image]]
    ) -> Optional[torch.Tensor]:
        if not images:
            return None
        pil_images = [self._to_pil(img) for img in images if img is not None]
        if not pil_images:
            return None
        # 与 DeQA / Q-Scorer 保持一致：先 resize / pad 再走 CLIP 预处理
        processed = process_images(pil_images, self.image_processor, model_cfg=self.model.config)
        if isinstance(processed, list):
            processed = torch.stack(processed, dim=0)
        return processed.to(self.model.device, dtype=torch.float16)

    def _build_prompt(
        self,
        prompt: str,
        num_images: int,
        system_prompt: Optional[str] = None,
    ) -> str:
        """根据 mPLUG-Owl2 的对话模板构建带 ``<|image|>`` 占位符的 Prompt。"""
        conv = conv_templates[self.conv_template].copy()
        if system_prompt:
            conv.system = system_prompt

        # 拼接 N 个图像占位符 + 用户文本
        image_tokens = (DEFAULT_IMAGE_TOKEN + "\n") * num_images
        user_msg = f"{image_tokens}{prompt}" if num_images > 0 else prompt
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)  # 留空，让模型生成
        return conv.get_prompt()

    def _build_inputs(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        image_tensor = self._build_image_tensor(images)
        num_imgs = 0 if image_tensor is None else image_tensor.shape[0]
        full_prompt = self._build_prompt(prompt, num_imgs, system_prompt)

        input_ids = (
            tokenizer_image_token(
                full_prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        attention_mask = torch.ones_like(input_ids, device=self.model.device)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": image_tensor,
        }

    # ------------------------------------------------------------------
    # 公共推理接口（与 Qwen25VLModel 对齐）
    # ------------------------------------------------------------------
    def chat(
        self,
        text: str,
        image: Union[str, Image.Image, np.ndarray, List[Any], None] = None,
        **kwargs: Any,
    ) -> str:
        """对外统一调用接口：纯文本、单图+文本、多图+文本。"""
        images = []
        if image is not None:
            images = image if isinstance(image, list) else [image]
        response = self.inference(images=images, prompt=text, **kwargs)
        return response.text

    def inference(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> VQAResponse:
        if not self._initialized:
            self.load_model()

        inputs = self._build_inputs(images, prompt, system_prompt)

        # === 为什么手动解码而不是 self.model.generate() ===
        # mPLUG-Owl2 的 modeling 文件（models/deqa/model/modeling_mplug_owl2.py）
        # 是早期 transformers 时代写的：
        #   - MPLUGOwl2LlamaForCausalLM.forward_single() 的签名是固定 kwargs，
        #     不带 **kwargs，无法接受现代 generate() 注入的 cache_position /
        #     num_logits_to_keep / DynamicCache 等参数；
        #   - prepare_inputs_for_generation() 也假设 past_key_values 是
        #     legacy 的 tuple-of-tuples 格式（会取 [-1][-1].shape[-2]）。
        # 因此一旦调用 .generate()，会在 _sample -> forward -> forward_single
        # -> self.model(...) 链路上崩。
        #
        # 我们的做法：自己写一个最朴素的 greedy / 温度采样循环，
        # 全程只调 self.model(input_ids, images, past_key_values=...)，
        # 把可能的 DynamicCache 立刻转回 legacy tuple-of-tuples，
        # 这样既兼容 DeQA 原版 forward_single，又兼容新版 transformers。
        generated_ids = self._manual_decode(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_tensor=inputs["images"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return VQAResponse(
            text=response_text,
            confidence=1.0,
            raw_output={"generated_ids": generated_ids},
        )

    # ------------------------------------------------------------------
    # 手动 decode 循环（绕开 HF generate 与 DeQA forward_single 的兼容坑）
    # ------------------------------------------------------------------
    def _manual_decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_tensor: Optional[torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[int]:
        """直接调 self.model 拿 logits，自己 sample/argmax。

        **完全不用 KV cache** —— 每一步把新生成的 token 拼回 ``input_ids``
        重新做一次 full forward。这正是 DeQA-Score / Q-Scorer 的
        ``Scorer.forward`` 所采用的同款模式（只是它们只 decode 2 步，
        我们这里 decode 到 EOS 或 ``max_new_tokens``）。

        为什么不省事直接复用 KV cache？
            DeQA 仓库自带的 ``modeling_llama2.model_forward`` 在
            ``past_key_values is not None`` 分支里硬性访问 ``[0][0].shape[2]``。
            在新版 transformers / accelerate 组合下，调用方传 ``None`` 到
            这一层时偶尔会被中间层替换成 ``(None,)*num_layers``，触发
            ``TypeError: 'NoneType' object is not subscriptable``。
            虽然 ``_apply_pkv_compat_patch()`` 已经把这类边角情况兜回
            ``None``，但更稳的做法是干脆不让 KV cache 进入这条路径。

        性能权衡：
            每步要重跑 ViT + LLM，单次 chat 调用大约比 KV cache 慢 5-10x。
            slow 模式下"新维度生长"过程因此会慢一些，但维度一旦固化后续走
            fast 路径（只调 ``predict_probs``，本来就是单次 forward），就
            完全不受影响。
        """
        device = self.model.device
        eos_id = self.tokenizer.eos_token_id
        mask_dtype = attention_mask.dtype

        generated: List[int] = []
        current_ids = input_ids
        current_mask = attention_mask

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    images=image_tensor,
                    use_cache=False,
                )

                next_token_logits = outputs.logits[:, -1, :]
                next_token = self._sample_next_token(
                    next_token_logits, temperature=temperature, top_p=top_p
                )

                token_id = int(next_token.item())
                if eos_id is not None and token_id == eos_id:
                    break
                generated.append(token_id)

                # 拼接新 token 到序列尾，下一轮整段重跑
                new_tok = next_token.view(1, 1).to(device)
                current_ids = torch.cat([current_ids, new_tok], dim=1)
                current_mask = torch.cat(
                    [current_mask, torch.ones((1, 1), device=device, dtype=mask_dtype)],
                    dim=1,
                )

        return generated

    @staticmethod
    def _sample_next_token(
        next_token_logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """与 Qwen 包装器对齐：temperature=0 时贪心，>0 时温度 + top-p 采样。"""
        if temperature <= 0:
            return torch.argmax(next_token_logits, dim=-1)

        logits = next_token_logits.float() / max(temperature, 1e-5)

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # 把累计概率超过 top_p 的位置 mask 掉（保证至少留下排名第一的）
            sorted_mask = cumprobs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter_(
                -1, sorted_indices, sorted_logits
            )

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def predict_probs(
        self,
        text: str,
        image: Union[str, Image.Image, np.ndarray, List[Any], None] = None,
        candidates: List[str] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """获取候选 token (如 A/B/C) 的 softmax 概率分布。

        与 ``Qwen25VLModel.predict_probs`` 等价：取最后一个位置的 logits，
        在候选 token 上做归一化。这是 Track 1 / Track 2 双轨制评分的底层 API。
        """
        if not self._initialized:
            self.load_model()

        candidates = candidates or ["A", "B", "C"]

        images = []
        if image is not None:
            images = image if isinstance(image, list) else [image]

        inputs = self._build_inputs(images, text)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=inputs["images"],
            )
            last_logits = outputs["logits"][0, -1, :]

        # 取每个候选词的 token id。
        # mPLUG-Owl2 使用 LLaMA 分词器，"excellent" 这种词在句中实际生成的是 "▁excellent"。
        # 与 DeQA-Score 官方做法保持一致：调用带 BOS 的 tokenizer，取 input_ids[1]
        # （第 0 个是 BOS），这能保证拿到的是带前导空格 ▁ 的那一个 token。
        # 如果 fallback 也失败（极少见的多 sub-token 词），退化到 encode(...)[-1]。
        candidate_ids = []
        for c in candidates:
            ids = self.tokenizer(c).input_ids
            # 去 BOS（如果存在）
            if len(ids) > 1 and ids[0] == self.tokenizer.bos_token_id:
                ids = ids[1:]
            if len(ids) == 0:
                ids = self.tokenizer.encode(c, add_special_tokens=False)
            candidate_ids.append(ids[0])  # 第一个 content token —— 与 DeQA preferential_ids_ 一致

        relevant_logits = last_logits[candidate_ids]
        probs = torch.softmax(relevant_logits.float(), dim=-1).cpu().numpy()

        return {cand: float(p) for cand, p in zip(candidates, probs)}


def create_mplug_owl2_model(
    model_path: str = "MAGAer13/mplug-owl2-llama2-7b",
    **kwargs: Any,
) -> MPLUGOwl2Model:
    """工厂函数。"""
    return MPLUGOwl2Model(model_path=model_path, **kwargs)
