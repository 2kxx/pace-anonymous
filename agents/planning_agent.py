"""
Planning Agent - 规划代理
实现快慢思考(System 1 & System 2)机制的核心调度器
"""

import json
import re
import time
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image

from models.vqa_model import Qwen25VLModel, VQAResponse
from models.q_scorer_eval import QScorerWrapper
from models.deqa_score_eval import DeQAScoreWrapper
from tools.crop_zoom import CropZoomTool
from memory.visual_rag import VisualRAGMemory


class ThinkingMode(Enum):
    """思考模式枚举"""
    FAST = "fast"      # 快思考 - 已知维度，直接推理
    SLOW = "slow"      # 慢思考 - 新维度，需要生长


@dataclass
class TaskResult:
    """任务执行结果"""
    success: bool
    dimension: str
    mode: ThinkingMode
    score: Optional[float] = None
    reasoning: str = ""
    features: Dict = field(default_factory=dict)
    memory_updated: bool = False
    execution_time: float = 0.0
    error: Optional[str] = None
    sub_dimensions: List[str] = field(default_factory=list)
    reflection_process: List[Dict] = field(default_factory=list)
    generated_protocol: List[Dict] = field(default_factory=list)
    pixel_features: List[str] = field(default_factory=list)
    anchors: List[Dict] = field(default_factory=list)
    evaluation_method: str = ""


class PlanningAgent:
    """
    规划代理 - 系统中枢

    评估策略：
    - 总体质量维度（overall, quality等）-> 快思考（Q-Scorer综合评分）
    - 传统维度（sharpness, noise等）-> 快思考（Q-Scorer单维度评分）
    - 已学习维度 -> 快思考（记忆库锚点对比）
    - 新增维度 -> 慢思考（反思闭环+记忆固化）
    """

    # 本地维度列表（备用，确保memory未初始化时也能正确分类）
    LOCAL_OVERALL_QUALITY_DIMENSIONS = [
        'overall', 'overall_quality', 'quality', '综合质量', '整体质量'
    ]

    LOCAL_TRADITIONAL_DIMENSIONS = [
        'sharpness', 'noise', 'contrast', 'brightness',
        'colorfulness', 'compression_artifact', 'blur',
        'exposure', 'saturation'
    ]

    def __init__(
        self,
        vqa_model: Qwen25VLModel = None,
        q_scorer: QScorerWrapper = None,
        deqa_score: DeQAScoreWrapper = None,
        crop_zoom: CropZoomTool = None,
        memory: VisualRAGMemory = None,
        image_generator: Any = None,
        max_iterations: int = 3,
        quality_threshold: float = 0.8,
        dimension_threshold: float = 0.8,
        **kwargs
    ):
        self.vqa_model = vqa_model
        self.q_scorer = q_scorer
        self.deqa_score = deqa_score
        self.crop_zoom = crop_zoom or CropZoomTool()
        self.memory = memory
        self.image_generator = image_generator
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.dimension_threshold = dimension_threshold

        # 子Agent
        self._visualizer = None
        self._critic = None
        self._generator = None

    def initialize_sub_agents(self):
        """初始化子Agent"""
        from agents.visualizer_agent import VisualizerAgent
        from agents.critic_agent import CriticAgent
        from agents.generator_agent import GeneratorAgent

        self._visualizer = VisualizerAgent(vqa_model=self.vqa_model)
        self._critic = CriticAgent(vqa_model=self.vqa_model)
        self._generator = GeneratorAgent(
            vqa_model=self.vqa_model,
            image_generator=self.image_generator,
            interactive=True
        )

    def _is_overall_quality_dimension(self, dimension: str) -> bool:
        """检查是否为总体质量维度"""
        dim_lower = dimension.lower().strip()

        # 优先使用memory中的方法
        if self.memory and hasattr(self.memory, 'is_overall_quality_dimension'):
            return self.memory.is_overall_quality_dimension(dim_lower)

        # 使用本地列表
        return dim_lower in [d.lower() for d in self.LOCAL_OVERALL_QUALITY_DIMENSIONS]

    def _is_traditional_dimension(self, dimension: str) -> bool:
        """检查是否为传统维度"""
        dim_lower = dimension.lower().strip()

        # 优先使用memory中的方法
        if self.memory and hasattr(self.memory, 'is_traditional_dimension'):
            return self.memory.is_traditional_dimension(dim_lower)

        # 使用本地列表
        return dim_lower in [d.lower() for d in self.LOCAL_TRADITIONAL_DIMENSIONS]

    def classify_dimension(self, dimension: str) -> Tuple[ThinkingMode, float, str]:
        """
        分类维度类型

        Returns:
            (ThinkingMode, confidence, dimension_category)
            dimension_category: "traditional" | "overall_quality" | "learned" | "novel"
        """
        dim_lower = dimension.lower().strip()

        # 1. 最高优先级：总体质量维度
        if self._is_overall_quality_dimension(dim_lower):
            return ThinkingMode.FAST, 1.0, "overall_quality"

        # 2. 检查是否为传统维度
        if self._is_traditional_dimension(dim_lower):
            return ThinkingMode.FAST, 1.0, "traditional"

        # 3. 查询记忆库
        if self.memory:
            similarity, _ = self.memory.query_dimension(dim_lower)
            if similarity >= self.dimension_threshold:
                return ThinkingMode.FAST, similarity, "learned"

        # 4. 默认为慢思考
        return ThinkingMode.SLOW, 0.5, "novel"

    def evaluate(
        self,
        image: Any,
        dimension: str,
        force_mode: ThinkingMode = None,
        external_anchors: List[Dict] = None,
        visualizer_mode: str = "standard",
        **kwargs
    ) -> TaskResult:
        """主评估接口"""
        start_time = time.time()

        # 图像预处理
        if isinstance(image, str):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))

        mode, confidence, category = self.classify_dimension(dimension)

        if visualizer_mode == "context-aware" and force_mode != ThinkingMode.FAST:
            print(f"  [Strategy] 维度 '{dimension}' 虽然已知，但由于开启 'context-aware' 模式，启动慢思考进化。")
            mode = ThinkingMode.SLOW

        if force_mode == ThinkingMode.FAST:
            mode = ThinkingMode.FAST
            if category != "learned":
                category = "forced"
        elif force_mode == ThinkingMode.SLOW:
            mode = ThinkingMode.SLOW

        # 路由
        if mode == ThinkingMode.FAST:
            return self._fast_thinking(image, dimension, category, start_time)
        else:
            # 透传 visualizer_mode
            return self._slow_thinking(
                image, dimension, start_time,
                external_anchors=external_anchors,
                visualizer_mode=visualizer_mode,
                **kwargs
            )

    def _fast_thinking(
        self,
        image: np.ndarray,
        dimension: str,
        category: str,
        start_time: float
    ) -> TaskResult:
        """快思考链路"""
        print(f"  → 启动快思考链路 (System 1)")

        try:
            # 策略1 & 2: 总体质量维度 或 传统维度 -> 使用双模型集成评估
            if category in ["overall_quality", "traditional", "forced"]:
                s_q = None
                s_deqa = None
                expert_details = []

                # 1. 尝试获取 Q-Scorer 分数
                if self.q_scorer:
                    try:
                        s_q = self.q_scorer.score(image)
                        expert_details.append(f"Q-Scorer: {s_q:.2f}")
                    except Exception as e:
                        print(f"  │ [Warning] Q-Scorer 推理出错: {e}")

                # 2. 尝试获取 DeQA-Score 分s
                if self.deqa_score:
                    try:
                        s_deqa = self.deqa_score.score(image)
                        expert_details.append(f"DeQA-Score: {s_deqa:.2f}")
                    except Exception as e:
                        print(f"  │ [Warning] DeQA-Score 推理出错: {e}")

                # 3. 分数合并逻辑 (加权平均策略)
                if s_q is not None and s_deqa is not None:
                    # 双专家共识：取平均值
                    final_score = (s_q + s_deqa) / 2.0
                    method_str = "ensemble (Q-Scorer + DeQA)"
                    reasoning = f"集成双专家评估结果。({', '.join(expert_details)})"
                elif s_q is not None or s_deqa is not None:
                    # 单专家降级
                    final_score = s_q if s_q is not None else s_deqa
                    method_str = "single_expert_fallback"
                    reasoning = f"单专家降级评估结果。({', '.join(expert_details)})"
                else:
                    # 彻底失败兜底
                    final_score = 3.0
                    method_str = "fallback_default"
                    reasoning = "专家模型均不可用，使用默认中值。"

                return TaskResult(
                    success=True,
                    dimension=dimension,
                    mode=ThinkingMode.FAST,
                    score=round(final_score, 2),
                    reasoning=reasoning,
                    features={
                        "q_val": s_q,
                        "deqa_val": s_deqa,
                        "discrepancy": abs(s_q - s_deqa) if (s_q is not None and s_deqa is not None) else 0
                    },
                    memory_updated=False,
                    execution_time=time.time() - start_time,
                    evaluation_method=method_str
                )

            # 策略3: 已学习维度 -> 记忆库锚点对比 (保持原有逻辑)
            elif category == "learned":
                if self.memory:
                    # 1. 从记忆库检索完整的生长成果
                    similarity, mem_entry = self.memory.query_dimension(dimension)

                    if mem_entry:
                        print(f"  │ [Memory] 匹配到固化维度 '{mem_entry.dimension}' (相似度: {similarity:.2f})")

                        # 2. 直接执行该维度固化的协议和锚点 (跳过慢思考的迭代过程)
                        # 这就是“知识沉淀”后的高效应用
                        score, reasoning = self._execute_perception_protocol(
                            image=image,
                            protocol=mem_entry.rules,
                            anchors=mem_entry.anchors
                        )

                        return TaskResult(
                            success=True,
                            dimension=dimension,
                            mode=ThinkingMode.FAST,
                            score=score,
                            reasoning=f"基于固化知识库的快速推理。\n{reasoning}",
                            features={"memory_similarity": similarity, "anchors_used": len(mem_entry.anchors)},
                            execution_time=time.time() - start_time,
                            evaluation_method="solidified_knowledge_inference"
                        )

            # 兜底：如果走到这里说明 learned 维度查不到记忆，或 category 不在已知分支里
            # 之前缺失 return 会让方法隐式返回 None，触发上层 AttributeError
            print(f"  │ [Fallback] category={category} 未命中任何快思考策略，降级为慢思考。")
            return self._slow_thinking(image, dimension, start_time)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return TaskResult(
                success=False,
                dimension=dimension,
                mode=ThinkingMode.FAST,
                error=str(e),
                execution_time=time.time() - start_time,
                evaluation_method="error"
            )

    def _decompose_dimension(self, dimension: str) -> List[str]:
        """
        [Step 1] 语义深度拆解 (Recursive Decomposition)
        利用 LLM 将抽象维度拆解为可观测的子维度 (Sub-dimensions)
        """
        print(f"  │ [Decomposition] 正在对 '{dimension}' 进行语义拆解...")

        prompt = (
            f"I need to evaluate the visual quality dimension: '{dimension}'.\n"
            f"Please decompose this abstract concept into 3 to 5 distinct, visually observable sub-dimensions or low-level features.\n"
            f"Output JSON format: {{'sub_dimensions': ['feature1', 'feature2', ...]}}"
        )

        try:
            # 假设 vqa_model 支持纯文本对话或带图对话（此处不需要图）
            response_str = self.vqa_model.chat(image=None, text=prompt)
            # 简单的解析逻辑，实际可能需要更鲁棒的 JSON 解析
            if "```json" in response_str:
                response_str = response_str.split("```json")[1].split("```")[0]
            data = json.loads(response_str)
            sub_dims = data.get("sub_dimensions", [])
            print(f"  │   -> 拆解结果: {sub_dims}")
            return sub_dims
        except Exception as e:
            print(f"  │   -> 拆解失败，使用原维度: {e}")
            return [dimension]

    def _slow_thinking(
            self,
            image: np.ndarray,
            dimension: str,
            start_time: float,
            external_anchors: List[Dict] = None,
            visualizer_mode: str = "standard",
            **kwargs
    ) -> TaskResult:
        """慢思考链路 - 完整的感知生长闭环"""
        print(f"  → 启动慢思考链路 (System 2) - 模式: {visualizer_mode}")
        if self._visualizer is None: self.initialize_sub_agents()

        try:
            reference_rules = None
            if self.memory:
                _, mem_entry = self.memory.query_dimension(dimension)
                if mem_entry:
                    reference_rules = mem_entry.rules
                    print(f"  │ [Memory] 提取到维度 '{dimension}' 的现有协议作为进化参考。")

            # 1. 语义深度拆解
            sub_dimensions = self._decompose_dimension(dimension)

            # 2. 对抗式反思闭环 (Adversarial Refinement Loop)
            print(f"  ╔══════════════════════════════════════════════════════╗")
            print(f"  ║     对抗式反思闭环 (Visualizer ⇄ Critic)             ║")
            print(f"  ╚══════════════════════════════════════════════════════╝")

            reflection_history = []
            critic_feedback = ""
            current_sub_dims = sub_dimensions

            best_protocol = []
            best_score = -1.0
            best_pixel_features = set()

            for iteration in range(self.max_iterations):
                print(f"  ┌─ 迭代 {iteration + 1}/{self.max_iterations} ─────────────────────────────")

                # Visualizer
                viz_result = self._visualizer.visualize(
                    dimension=dimension,
                    sub_dimensions=current_sub_dims,
                    feedback=critic_feedback,
                    image=image,
                    crop_zoom=self.crop_zoom,
                    iteration=iteration + 1,
                    mode=visualizer_mode,
                    reference_rules=reference_rules
                )
                proposed_rules = viz_result.get("rules", [])

                # Critic
                critique = self._critic.critique(
                    dimension=dimension,
                    rules=proposed_rules,
                    image=image,
                    crop_zoom=self.crop_zoom
                )

                validated_rules = critique.get("validated_rules", [])
                rejected_reasons = critique.get("rejected_reasons", [])
                quality_score = critique.get("quality_score", 0.0)
                pixel_features = critique.get("pixel_features", [])

                # 记录反思历史
                reflection_history.append({
                    "iteration": iteration + 1,
                    "proposed": len(proposed_rules),
                    "rules_validated": len(validated_rules),
                    "quality_score": quality_score
                })

                print(
                    f"  │   Visualizer 提出 {len(proposed_rules)} 条 -> Critic 通过 {len(validated_rules)} 条 (质量={quality_score:.2f})")

                if quality_score > best_score and len(validated_rules) > 0:
                    best_score = quality_score
                    best_protocol = validated_rules
                    best_pixel_features = set(pixel_features)

                # 如果达标，直接提前跳出
                if quality_score >= self.quality_threshold:
                    print(f"  │ ✓ 协议质量达标，结束博弈")
                    break

                # 如果没达标，构建反馈给下一轮
                critic_feedback = f"Some rules were rejected because: {'; '.join(rejected_reasons)}. Please fix them and ensure they are visually observable."

            final_protocol = best_protocol
            all_pixel_features = best_pixel_features

            # 3. 生成增强 (Generative Augmentation)
            # 利用协议反向合成 Anchor，作为“视觉教具”
            print(f"\n  [Phase 3] 生成增强与锚点构建")
            anchors = []
            if external_anchors:
                anchors = external_anchors
                print(f"    使用外部提供的 {len(anchors)} 个 Benchmark 真实锚点 (跳过 SD 生成)")
            elif self._generator and final_protocol:
                gen_result = self._generator.generate_anchors(dimension=dimension, rules=final_protocol)
                anchors = gen_result.get("anchors", [])
                print(f"    合成 {len(anchors)} 个视觉锚点样本")

            # 4. 执行感知协议 (Execution)
            # 这是一个关键步骤：不再是瞎猜分数，而是运行刚才生成的 VQA 问题
            print(f"\n  [Phase 4] 执行感知协议 (Inference)")
            final_score, reasoning_trace = self._execute_perception_protocol(
                image=image,
                protocol=final_protocol,
                anchors=anchors
            )
            print(f"    推理完成: Score = {final_score}")

            # 5. 能力固化 (Memory Solidification)
            memory_updated = False
            if self.memory and final_protocol:
                self.memory.add_dimension_entry(
                    dimension=dimension,
                    sub_dimensions=sub_dimensions,
                    rules=final_protocol,
                    pixel_features=list(all_pixel_features),
                    anchors=anchors
                )
                memory_updated = True
                print(f"    ✓ 维度 '{dimension}' 已固化为已知能力")

            return TaskResult(
                success=True,
                dimension=dimension,
                mode=ThinkingMode.SLOW,
                score=final_score,
                reasoning=reasoning_trace,
                sub_dimensions=sub_dimensions,
                generated_protocol=final_protocol,
                reflection_process=reflection_history,
                features={"pixel_features": list(all_pixel_features)},
                anchors=anchors,
                memory_updated=memory_updated,
                execution_time=time.time() - start_time,
                evaluation_method="generated_protocol_inference"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return TaskResult(
                success=False,
                dimension=dimension,
                mode=ThinkingMode.SLOW,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _execute_perception_protocol(
            self,
            image: np.ndarray,
            protocol: List[Dict],
            anchors: List[Dict]
    ) -> Tuple[float, str]:
        """
        [Step 4 Implementation]
        双轨制评估架构 (Dual-Track Scoring):
        Track 1: 绝对感知评分 (基于工具辅助的 VQA 客观问答)
        Track 2: 人类偏好评分 (基于生成式 Anchors 的宏观对比)
        """
        import os
        from PIL import Image

        if not protocol:
            return 3.0, "未能生成有效感知协议，默认评分。"

        evidence_chain = []

        # =================================================================
        # Track 1: 绝对感知评分 (Absolute Perceptive Score)
        # =================================================================
        track1_score = 0.0
        weight_sum = 0.0
        micro_keywords = ['detail', 'edge', 'texture', 'noise', 'sharpness', 'grain', 'pixel', 'blur']

        evidence_chain.append("【Track 1: 绝对感知特征评估 (Logits Softmax)】")
        for rule in protocol:
            question = rule.get("question", "")
            options = rule.get("option_scores", {})  # 原本是 {"Very clear": 5, "Blurry": 1}
            w = rule.get("weight", 1.0)

            # --- 核心改进：映射动态选项到字母 ---
            # 按分数从高到低排序，确保 A 是最高分
            sorted_opts = sorted(options.items(), key=lambda x: x[1], reverse=True)
            labels = ["A", "B", "C", "D", "E"][:len(sorted_opts)]

            # 构建 Prompt 里的选项列表
            # 结果示例: "A) Very clear, B) Blurry"
            options_text_list = [f"{labels[i]}) {sorted_opts[i][0]}" for i in range(len(sorted_opts))]
            options_list_str = "\n".join(options_text_list)

            images_to_feed = [image]

            # 触发放大镜
            needs_zoom = any(kw in question.lower() for kw in micro_keywords)
            if needs_zoom and self.crop_zoom is not None:
                try:
                    patch = self.crop_zoom.process(image)
                    images_to_feed.append(patch)
                    evidence_chain.append(f"  * 工具触发: 针对问题 '{question[:15]}...' 调用 CropZoom")
                except Exception:
                    pass

            full_prompt = (
                f"TASK: Evaluate the image based on the following criteria.\n"
                f"QUESTION: {question}\n\n"
                f"OPTIONS:\n{options_list_str}\n\n"
                f"Select the most accurate option. REPLY WITH ONLY THE LETTER (A, B, or C)."
            )

            try:
                # 调用 predict_probs 获取 A, B, C 的概率
                probs_dict = self.vqa_model.predict_probs(
                    text=full_prompt,
                    image=images_to_feed,
                    candidates=labels
                )

                # 计算期望分数: Prob(A)*Score(A) + Prob(B)*Score(B) ...
                rule_score = sum(probs_dict[labels[i]] * sorted_opts[i][1] for i in range(len(sorted_opts)))

                track1_score += rule_score * w
                weight_sum += w

                # 记录精细的 Logits 分布情况
                prob_info = ", ".join([f"{labels[i]}:{probs_dict[labels[i]]:.2f}" for i in range(len(labels))])
                evidence_chain.append(f"  - Q: {question[:25]}... \n    Logits: {prob_info} -> 连续得分: {rule_score:.3f}")

            except Exception as e:
                print(f"  │ [Logits Error] {e}")
                # 失败兜底
                track1_score += 3.0 * w
                weight_sum += w

        absolute_score = round(track1_score / weight_sum, 2) if weight_sum > 0 else 3.0
        alpha_logic = (absolute_score - 1.0) / 4.0
        evidence_chain.append(f"  => Track 1 感知强度: {alpha_logic:.4f} (Score: {absolute_score:.2f})")

        # =================================================================
        # Track 2: 尺度校准 (Logits 相对位置)
        # =================================================================
        alpha_scale = alpha_logic  # 默认等于 Track 1
        track2_executed = False

        if anchors and len(anchors) >= 2:
            evidence_chain.append("【Track 2: 锚点对比 Logits 校准】")
            sorted_anchors = sorted(anchors, key=lambda x: x.get('score', 0), reverse=True)
            high_a, low_a = sorted_anchors[0], sorted_anchors[-1]

            if os.path.exists(high_a["image_path"]) and os.path.exists(low_a["image_path"]):
                anchor_imgs = [Image.open(high_a["image_path"]).convert('RGB'),
                               Image.open(low_a["image_path"]).convert('RGB'),
                               image]

                compare_labels = ["A", "B", "C", "D", "E"]
                compare_weights = [1.0, 0.75, 0.5, 0.25, 0.0]

                prompt_t2 = (
                    f"Reference Image 1: High Quality\n"
                    f"Reference Image 2: Low Quality\n"
                    f"Target Image: Image 3\n\n"
                    f"Which reference image is Image 3 more similar to in terms of quality?\n"
                    f"A) Extremely similar to Image 1\n"
                    f"B) Closer to Image 1\n"
                    f"C) Right in the middle\n"
                    f"D) Closer to Image 2\n"
                    f"E) Extremely similar to Image 2\n"
                    f"REPLY WITH THE LETTER ONLY."
                )

                try:
                    # 获取 A-E 的 Logits 概率分布
                    probs_scale = self.vqa_model.predict_probs(text=prompt_t2, image=anchor_imgs,
                                                               candidates=compare_labels)

                    # 计算概率重心作为“相对位置系数” (0.0 - 1.0 连续浮点数)
                    alpha_scale = sum(probs_scale[l] * compare_weights[i] for i, l in enumerate(compare_labels))
                    track2_executed = True

                    dist_info = ", ".join([f"{l}:{probs_scale[l]:.2f}" for l in compare_labels])
                    evidence_chain.append(f"  - 相似度分布: {dist_info}")
                    evidence_chain.append(f"  - 概率重心位置 (alpha_scale): {alpha_scale:.4f}")
                except:
                    pass

        # =================================================================
        # 融合与放缩结算 (Scaling)
        # =================================================================
        if track2_executed:
            combined_alpha = (alpha_logic * 0.5) + (alpha_scale * 0.5)

            # 使用锚点的真实分值范围进行线性拉伸
            high_gt = sorted_anchors[0]['score']
            low_gt = sorted_anchors[-1]['score']

            final_score = low_gt + (high_gt - low_gt) * combined_alpha

            evidence_chain.append(f"  - 锚点量程: [{low_gt:.2f}, {high_gt:.2f}]")
            evidence_chain.append(f"  - 最终放缩分: {final_score:.3f}")
        else:
            final_score = absolute_score

        return round(max(1.0, min(5.0, final_score)), 3), "\n".join(evidence_chain)
