"""模型模块"""

from .vqa_model import Qwen25VLModel, VQAResponse, create_vqa_model
from .q_scorer_eval import QScorerWrapper, create_q_scorer

# mPLUG-Owl2 是可选骨架（依赖 peft/icecream 等），延迟导入避免缺包时直接 ImportError
try:
    from .mplug_owl2_model import MPLUGOwl2Model, create_mplug_owl2_model  # noqa: F401
except Exception as _exc:  # noqa: BLE001
    MPLUGOwl2Model = None  # type: ignore[assignment]
    create_mplug_owl2_model = None  # type: ignore[assignment]
    _MPLUG_IMPORT_ERROR = _exc

__all__ = [
    'Qwen25VLModel', 'VQAResponse', 'create_vqa_model',
    'QScorerWrapper', 'create_q_scorer',
    'MPLUGOwl2Model', 'create_mplug_owl2_model',
]
