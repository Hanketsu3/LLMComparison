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
    "qwen3-vl-2b": ModelInfo(
        name="qwen3-vl-2b",
        display_name="Qwen3-VL-2B",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="2B",
        description="Qwen3-VL Instruct model for compact multimodal reasoning",
        colab_t4_native=False,
        needs_4bit=True,
        default_kwargs={"model_name": "Qwen/Qwen3-VL-2B-Instruct", "load_in_4bit": True},
    ),
    "qwen2.5-vl-3b": ModelInfo(
        name="qwen2.5-vl-3b",
        display_name="Qwen2.5-VL-3B",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="3B",
        description="Qwen2.5-VL 3B model for stronger lightweight multimodal performance",
        colab_t4_native=False,
        needs_4bit=True,
        default_kwargs={"model_name": "Qwen/Qwen2.5-VL-3B-Instruct", "load_in_4bit": True},
    ),
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
    "phi3-vision": ModelInfo(
        name="phi3-vision",
        display_name="Phi-3.5-Vision",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.phi3_vision",
        class_name="Phi3VisionModel",
        params="4.2B",
        description="Microsoft's compact VLM (Phi-3.5 updated)",
        colab_t4_native=True,
        default_kwargs={"model_name": "microsoft/Phi-3.5-vision-instruct"},
    ),
    "smolvlm2-2.2b": ModelInfo(
        name="smolvlm2-2.2b",
        display_name="SmolVLM2-2.2B",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="2.2B",
        description="SmolVLM2 lightweight multimodal model for Colab-scale experiments",
        colab_t4_native=True,
        default_kwargs={"model_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "load_in_4bit": False},
    ),
    "llama3-vision": ModelInfo(
        name="llama3-vision",
        display_name="Llama-3.2 Vision",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.llama3",
        class_name="Llama3Model",
        params="11B",
        description="Meta's multimodal Llama-3.2 Vision (gated access enabled)",
        colab_t4_native=False,
        needs_4bit=True,
        default_kwargs={"model_name": "meta-llama/Llama-3.2-11B-Vision-Instruct", "load_in_4bit": True},
    ),
    
    # --- Domain-Adaptive (Free) ---
    "qwen2-vl-ocr-2b": ModelInfo(
        name="qwen2-vl-ocr-2b",
        display_name="Qwen2-VL-OCR-2B",
        category=ModelCategory.DOMAIN_ADAPTIVE,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.qwen2_vl",
        class_name="Qwen2VLModel",
        params="2B",
        description="Qwen2-VL 2B fine-tuned for OCR and document-style visual understanding",
        colab_t4_native=True,
        default_kwargs={"model_name": "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"},
    ),
    "latxa-qwen3-vl-2b": ModelInfo(
        name="latxa-qwen3-vl-2b",
        display_name="Latxa-Qwen3-VL-2B",
        category=ModelCategory.DOMAIN_ADAPTIVE,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="2B",
        description="Language-adapted Qwen3-VL for Basque/Galician/Catalan use-cases",
        colab_t4_native=False,
        needs_4bit=True,
        default_kwargs={"model_name": "HiTZ/Latxa-Qwen3-VL-2B-Instruct", "load_in_4bit": True},
    ),
    "medgemma-4b": ModelInfo(
        name="medgemma-4b",
        display_name="MedGemma-4B-IT",
        category=ModelCategory.DOMAIN_ADAPTIVE,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="4B",
        description="Google MedGemma model adapted for medical text-image understanding",
        colab_t4_native=False,
        needs_4bit=True,
        default_kwargs={"model_name": "google/medgemma-4b-it", "load_in_4bit": True},
    ),
    "llava-med": ModelInfo(
        name="llava-med",
        display_name="LLaVA-Med",
        category=ModelCategory.DOMAIN_ADAPTIVE,
        cost=ModelCost.FREE,
        module_path="src.models.domain_adaptive.llava_med",
        class_name="LLaVAMedModel",
        params="7B",
        description="LLaVA fine-tuned on biomedical data (Vicuna-7B variant)",
        colab_t4_native=False,
        needs_4bit=True,
        default_kwargs={"model_name": "microsoft/llava-med-v1.5-vicuna-7b", "load_in_4bit": True},
    ),
    # Legacy domain-adaptive options can still be evaluated if needed.
    
    # --- Specialist (Free) ---
    "got-ocr2": ModelInfo(
        name="got-ocr2",
        display_name="GOT-OCR-2.0",
        category=ModelCategory.SPECIALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="0.58B",
        description="OCR-specialist multimodal model for documents, formulas, and scene text",
        colab_t4_native=True,
        default_kwargs={"model_name": "stepfun-ai/GOT-OCR-2.0-hf", "load_in_4bit": False},
    ),
    "nougat-base": ModelInfo(
        name="nougat-base",
        display_name="Nougat-Base",
        category=ModelCategory.SPECIALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="base",
        description="Scientific PDF-to-markdown specialist model",
        colab_t4_native=True,
        default_kwargs={"model_name": "facebook/nougat-base", "load_in_4bit": False},
    ),
    "matcha-chartqa": ModelInfo(
        name="matcha-chartqa",
        display_name="MatCha-ChartQA",
        category=ModelCategory.SPECIALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="base",
        description="ChartQA-specialist model for chart and plot reasoning",
        colab_t4_native=True,
        default_kwargs={"model_name": "google/matcha-chartqa", "load_in_4bit": False},
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
