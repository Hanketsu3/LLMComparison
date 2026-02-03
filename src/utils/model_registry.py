"""
Model Registry - Organize models by category and cost

Provides easy access to models with clear cost indicators.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from enum import Enum


class ModelCost(Enum):
    """Model cost category."""
    FREE = "free"      # Completely free, open-source
    PAID = "paid"      # Requires paid API access


class ModelCategory(Enum):
    """Model category based on training approach."""
    GENERALIST = "generalist"           # General-purpose VLMs
    DOMAIN_ADAPTIVE = "domain_adaptive" # Medical domain adapted
    SPECIALIST = "specialist"           # Radiology-specific


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    display_name: str
    category: ModelCategory
    cost: ModelCost
    module_path: str
    class_name: str
    params: str  # Parameter count (e.g., "7B", "2B")
    description: str
    colab_t4_native: bool  # Runs on Colab T4 without quantization
    needs_4bit: bool = False  # Requires 4-bit quantization for T4
    supports_grounding: bool = False
    default_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.default_kwargs is None:
            self.default_kwargs = {}
    
    @property
    def colab_compatible(self) -> bool:
        """Can run on Colab T4 (native or with 4-bit)."""
        return self.colab_t4_native or self.needs_4bit


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # =========================================================================
    # 🆓 FREE MODELS - Open Source
    # =========================================================================
    
    # --- Generalist (Free) ---
    "qwen2-vl-2b": ModelInfo(
        name="qwen2-vl-2b",
        display_name="Qwen2-VL-2B",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.qwen2_vl",
        class_name="Qwen2VLModel",
        params="2B",
        description="Alibaba's efficient vision-language model",
        colab_t4_native=True,
        default_kwargs={"model_name": "Qwen/Qwen2-VL-2B-Instruct"},
    ),
    "qwen2-vl-7b": ModelInfo(
        name="qwen2-vl-7b",
        display_name="Qwen2-VL-7B",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.qwen2_vl",
        class_name="Qwen2VLModel",
        params="7B",
        description="Alibaba's larger vision-language model",
        colab_t4_native=False,
        needs_4bit=True,
        default_kwargs={"model_name": "Qwen/Qwen2-VL-7B-Instruct", "load_in_4bit": True},
    ),
    "phi3-vision": ModelInfo(
        name="phi3-vision",
        display_name="Phi-3 Vision",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.phi3_vision",
        class_name="Phi3VisionModel",
        params="4B",
        description="Microsoft's small but powerful VLM",
        colab_t4_native=True,
    ),
    "internvl2-2b": ModelInfo(
        name="internvl2-2b",
        display_name="InternVL2-2B",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.internvl2",
        class_name="InternVL2Model",
        params="2B",
        description="OpenGVLab's efficient VLM",
        colab_t4_native=True,
        default_kwargs={"model_name": "OpenGVLab/InternVL2-2B"},
    ),
    "internvl2-8b": ModelInfo(
        name="internvl2-8b",
        display_name="InternVL2-8B",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.internvl2",
        class_name="InternVL2Model",
        params="8B",
        description="OpenGVLab's larger VLM",
        colab_t4_native=False,
        needs_4bit=True,
        default_kwargs={"model_name": "OpenGVLab/InternVL2-8B", "load_in_4bit": True},
    ),
    "minicpm-v": ModelInfo(
        name="minicpm-v",
        display_name="MiniCPM-V-2.6",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.minicpm_v",
        class_name="MiniCPMVModel",
        params="8B",
        description="OpenBMB's efficient multimodal LLM",
        colab_t4_native=True,
    ),
    "llava-next-7b": ModelInfo(
        name="llava-next-7b",
        display_name="LLaVA-NeXT-7B",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.llava_next",
        class_name="LLaVANextModel",
        params="7B",
        description="Open-source successor to LLaVA",
        colab_t4_native=False,
        needs_4bit=True,
        default_kwargs={"load_in_4bit": True},
    ),
    "llama3-vision": ModelInfo(
        name="llama3-vision",
        display_name="Llama-3 Vision",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.llama3",
        class_name="Llama3Model",
        params="8B",
        description="Meta's open-source LLM with vision",
        colab_t4_native=False,
        needs_4bit=True,
    ),
    
    # --- Domain-Adaptive (Free) ---
    "llava-med": ModelInfo(
        name="llava-med",
        display_name="LLaVA-Med",
        category=ModelCategory.DOMAIN_ADAPTIVE,
        cost=ModelCost.FREE,
        module_path="src.models.domain_adaptive.llava_med",
        class_name="LLaVAMedModel",
        params="7B",
        description="LLaVA fine-tuned on biomedical data",
        colab_t4_native=False,
        needs_4bit=True,
        default_kwargs={"load_in_4bit": True},
    ),
    "biomedgpt": ModelInfo(
        name="biomedgpt",
        display_name="BiomedGPT",
        category=ModelCategory.DOMAIN_ADAPTIVE,
        cost=ModelCost.FREE,
        module_path="src.models.domain_adaptive.biomedgpt",
        class_name="BiomedGPTModel",
        params="7B",
        description="Biomedical domain LLM",
        colab_t4_native=False,
        needs_4bit=True,
    ),
    
    # --- Specialist (Free) ---
    "chexagent": ModelInfo(
        name="chexagent",
        display_name="CheXagent",
        category=ModelCategory.SPECIALIST,
        cost=ModelCost.FREE,
        module_path="src.models.specialist.chexagent",
        class_name="CheXagentModel",
        params="8B",
        description="Stanford's chest X-ray specialist",
        colab_t4_native=False,
        needs_4bit=True,
        supports_grounding=True,
        default_kwargs={"load_in_4bit": True},
    ),
    "llava-rad": ModelInfo(
        name="llava-rad",
        display_name="LLaVA-Rad",
        category=ModelCategory.SPECIALIST,
        cost=ModelCost.FREE,
        module_path="src.models.specialist.llava_rad",
        class_name="LLaVARadModel",
        params="7B",
        description="Radiology-specialized LLaVA",
        colab_t4_native=False,
        needs_4bit=True,
        default_kwargs={"load_in_4bit": True},
    ),
    "radfm": ModelInfo(
        name="radfm",
        display_name="RadFM",
        category=ModelCategory.SPECIALIST,
        cost=ModelCost.FREE,
        module_path="src.models.specialist.radfm",
        class_name="RadFMModel",
        params="varies",
        description="Radiology Foundation Model",
        colab_t4_native=True,
    ),
    
    # =========================================================================
    # 💰 PAID MODELS - Require API Access
    # =========================================================================
    
    "gpt4v": ModelInfo(
        name="gpt4v",
        display_name="GPT-4 Vision",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.PAID,
        module_path="src.models.generalist.gpt4v",
        class_name="GPT4VModel",
        params="unknown",
        description="OpenAI's GPT-4 with vision (API)",
        colab_t4_native=True,  # API-based, no GPU needed
    ),
    "gemini": ModelInfo(
        name="gemini",
        display_name="Gemini Pro Vision",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.PAID,
        module_path="src.models.generalist.gemini",
        class_name="GeminiModel",
        params="unknown",
        description="Google's Gemini with vision (API)",
        colab_t4_native=True,  # API-based
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_free_models() -> List[ModelInfo]:
    """Get all free/open-source models."""
    return [m for m in MODEL_REGISTRY.values() if m.cost == ModelCost.FREE]


def get_paid_models() -> List[ModelInfo]:
    """Get all paid API models."""
    return [m for m in MODEL_REGISTRY.values() if m.cost == ModelCost.PAID]


def get_colab_compatible_models() -> List[ModelInfo]:
    """Get models that can run on Colab T4 (16GB)."""
    return [m for m in MODEL_REGISTRY.values() if m.colab_compatible]


def get_models_by_category(category: ModelCategory) -> List[ModelInfo]:
    """Get models by category."""
    return [m for m in MODEL_REGISTRY.values() if m.category == category]


def get_model_info(name: str) -> Optional[ModelInfo]:
    """Get model info by name."""
    return MODEL_REGISTRY.get(name)


def load_model(name: str):
    """Dynamically load and instantiate a model."""
    info = get_model_info(name)
    if info is None:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    import importlib
    module = importlib.import_module(info.module_path)
    model_class = getattr(module, info.class_name)
    
    return model_class(**info.default_kwargs)


def print_model_table():
    """Print a formatted table of all models."""
    print("\n" + "="*80)
    print("🆓 FREE MODELS (Open Source)")
    print("="*80)
    print(f"{'Name':<20} {'Category':<15} {'Params':<8} {'T4 GPU':<12} Description")
    print("-"*80)
    
    for model in get_free_models():
        if model.colab_t4_native:
            t4_status = "✅ Native"
        elif model.needs_4bit:
            t4_status = "✅ 4-bit"
        else:
            t4_status = "❌"
        print(f"{model.display_name:<20} {model.category.value:<15} {model.params:<8} {t4_status:<12} {model.description[:25]}")
    
    print("\n" + "="*80)
    print("💰 PAID MODELS (API Required)")
    print("="*80)
    print(f"{'Name':<20} {'Category':<15} {'Params':<8} {'T4 GPU':<12} Description")
    print("-"*80)
    
    for model in get_paid_models():
        print(f"{model.display_name:<20} {model.category.value:<15} {model.params:<8} {'✅ API':<12} {model.description[:25]}")


# Quick reference lists
FREE_MODELS = [m.name for m in get_free_models()]
PAID_MODELS = [m.name for m in get_paid_models()]
COLAB_MODELS = [m.name for m in get_colab_compatible_models()]

# Category-specific lists
GENERALIST_MODELS = [m.name for m in get_models_by_category(ModelCategory.GENERALIST)]
DOMAIN_ADAPTIVE_MODELS = [m.name for m in get_models_by_category(ModelCategory.DOMAIN_ADAPTIVE)]
SPECIALIST_MODELS = [m.name for m in get_models_by_category(ModelCategory.SPECIALIST)]
