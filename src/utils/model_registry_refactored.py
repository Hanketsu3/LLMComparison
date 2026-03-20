"""
Model Registry - Organize models by category and benchmark tier

Synced with src/configs/model_taxonomy.py:
- MAIN benchmark (13 models): radiology-focused
- EXTRA track (8 models): out-of-scope experimental track
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
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


class BenchmarkTier(Enum):
    """Benchmark tier classification."""
    MAIN = "main"      # Core radiology comparison (13 models)
    EXTRA = "extra"    # Out-of-scope experimental track (8 models)


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
    benchmark_tier: BenchmarkTier = BenchmarkTier.MAIN  # NEW: Tier classification
    needs_4bit: bool = False  # Requires 4-bit quantization for T4
    supports_grounding: bool = False
    gated_access: bool = False  # Requires HF gated access approval
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def colab_compatible(self) -> bool:
        """Can run on Colab T4 (native or with 4-bit)."""
        return self.colab_t4_native or self.needs_4bit


# ============================================================================
# MODEL REGISTRY - MAIN BENCHMARK (13 Models)
# ============================================================================

MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # =========================================================================
    # GENERALIST (6 models) - General-purpose, lightweight
    # =========================================================================
    
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
        benchmark_tier=BenchmarkTier.MAIN,
        default_kwargs={"model_name": "Qwen/Qwen2-VL-2B-Instruct"},
    ),
    "qwen2.5-vl-3b": ModelInfo(
        name="qwen2.5-vl-3b",
        display_name="Qwen2.5-VL-3B",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="3B",
        description="Improved Qwen with better instruction following",
        colab_t4_native=False,
        benchmark_tier=BenchmarkTier.MAIN,
        needs_4bit=True,
        default_kwargs={"model_name": "Qwen/Qwen2.5-VL-3B-Instruct", "load_in_4bit": True},
    ),
    "qwen3-vl-2b": ModelInfo(
        name="qwen3-vl-2b",
        display_name="Qwen3-VL-2B",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="2B",
        description="Latest Qwen series, compact multimodal reasoning",
        colab_t4_native=False,
        benchmark_tier=BenchmarkTier.MAIN,
        needs_4bit=True,
        default_kwargs={"model_name": "Qwen/Qwen3-VL-2B-Instruct", "load_in_4bit": True},
    ),
    "phi3-vision": ModelInfo(
        name="phi3-vision",
        display_name="Phi-3.5-Vision",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.phi3_vision",
        class_name="Phi3VisionModel",
        params="4.2B",
        description="Microsoft's compact VLM with good quality-to-size ratio",
        colab_t4_native=True,
        benchmark_tier=BenchmarkTier.MAIN,
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
        description="Hugging Face lightweight model optimized for resource-constrained settings",
        colab_t4_native=True,
        benchmark_tier=BenchmarkTier.MAIN,
        default_kwargs={"model_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"},
    ),
    "llama3-vision": ModelInfo(
        name="llama3-vision",
        display_name="Llama-3.2-Vision",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.llama3",
        class_name="Llama3Model",
        params="11B",
        description="Meta's multimodal Llama-3.2 Vision (gated access)",
        colab_t4_native=False,
        benchmark_tier=BenchmarkTier.MAIN,
        needs_4bit=True,
        gated_access=True,
        default_kwargs={"model_name": "meta-llama/Llama-3.2-11B-Vision-Instruct", "load_in_4bit": True},
    ),
    
    # =========================================================================
    # DOMAIN-ADAPTIVE (3 models) - Medical/biomedical specialized
    # =========================================================================
    
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
        benchmark_tier=BenchmarkTier.MAIN,
        needs_4bit=True,
        default_kwargs={"model_name": "microsoft/llava-med-v1.5-vicuna-7b", "load_in_4bit": True},
    ),
    "medgemma-4b": ModelInfo(
        name="medgemma-4b",
        display_name="MedGemma-4B",
        category=ModelCategory.DOMAIN_ADAPTIVE,
        cost=ModelCost.FREE,
        module_path="src.models.domain_adaptive.biomedgpt",  # TBD: actual module
        class_name="GenericVisionChatModel",
        params="4B",
        description="Google's medical domain-adapted Gemma variant",
        colab_t4_native=False,
        benchmark_tier=BenchmarkTier.MAIN,
        needs_4bit=True,
        default_kwargs={"model_name": "google/medgemma-4b-it", "load_in_4bit": True},
    ),
    "biomedgpt": ModelInfo(
        name="biomedgpt",
        display_name="BiomedGPT",
        category=ModelCategory.DOMAIN_ADAPTIVE,
        cost=ModelCost.FREE,
        module_path="src.models.domain_adaptive.biomedgpt",
        class_name="BiomedGPTModel",
        params="7B",
        description="Medical text corpus pre-trained variant",
        colab_t4_native=False,
        benchmark_tier=BenchmarkTier.MAIN,
        needs_4bit=True,
        default_kwargs={"model_name": "PharMolix/BiomedGPT-LM-7B"},
    ),
    
    # =========================================================================
    # SPECIALIST (3 models) - Radiology-specific
    # =========================================================================
    
    "chexagent": ModelInfo(
        name="chexagent",
        display_name="CheXagent",
        category=ModelCategory.SPECIALIST,
        cost=ModelCost.FREE,
        module_path="src.models.specialist.chexagent",
        class_name="CheXagentModel",
        params="8B",
        description="Stanford AIMI chest X-ray expert with phrase grounding support",
        colab_t4_native=False,
        benchmark_tier=BenchmarkTier.MAIN,
        needs_4bit=True,
        supports_grounding=True,
        gated_access=True,
        default_kwargs={"model_name": "StanfordAIMI/CheXagent-8b"},
    ),
    "llava-rad": ModelInfo(
        name="llava-rad",
        display_name="LLaVA-Rad",
        category=ModelCategory.SPECIALIST,
        cost=ModelCost.FREE,
        module_path="src.models.specialist.llava_rad",
        class_name="LLaVARadModel",
        params="8B",
        description="LLaVA fine-tuned on radiology reports and images",
        colab_t4_native=False,
        benchmark_tier=BenchmarkTier.MAIN,
        needs_4bit=True,
        default_kwargs={"model_name": "microsoft/llava-rad", "load_in_4bit": True},
    ),
    "radfm": ModelInfo(
        name="radfm",
        display_name="RadFM",
        category=ModelCategory.SPECIALIST,
        cost=ModelCost.FREE,
        module_path="src.models.specialist.radfm",
        class_name="RadFMModel",
        params="7B",
        description="Foundation model trained on radiology images and reports",
        colab_t4_native=False,
        benchmark_tier=BenchmarkTier.MAIN,
        needs_4bit=True,
        default_kwargs={"model_name": "StanfordAIMI/RadFM-7B"},
    ),
    
    # =========================================================================
    # EXTRA TRACK (8 Models) - Out-of-scope experimental
    # =========================================================================
    
    # --- OCR / Document Analysis ---
    "got-ocr2": ModelInfo(
        name="got-ocr2",
        display_name="GOT-OCR-2.0",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="0.58B",
        description="OCR specialist for documents, formulas, scene text",
        colab_t4_native=True,
        benchmark_tier=BenchmarkTier.EXTRA,
        default_kwargs={"model_name": "stepfun-ai/GOT-OCR-2.0-hf"},
    ),
    "nougat-base": ModelInfo(
        name="nougat-base",
        display_name="Nougat",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="0.35B",
        description="Scientific PDF-to-markdown specialist model",
        colab_t4_native=True,
        benchmark_tier=BenchmarkTier.EXTRA,
        default_kwargs={"model_name": "facebook/nougat-base"},
    ),
    "matcha-chartqa": ModelInfo(
        name="matcha-chartqa",
        display_name="MatCha-ChartQA",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.generic_vision_chat",
        class_name="GenericVisionChatModel",
        params="0.67B",
        description="Chart and plot reasoning specialist",
        colab_t4_native=True,
        benchmark_tier=BenchmarkTier.EXTRA,
        default_kwargs={"model_name": "google/matcha-chartqa"},
    ),
    
    # --- Language-Adapted ---
    "qwen2-vl-ocr": ModelInfo(
        name="qwen2-vl-ocr",
        display_name="Qwen2-VL-OCR",
        category=ModelCategory.DOMAIN_ADAPTIVE,
        cost=ModelCost.FREE,
        module_path="src.models.generalist.qwen2_vl",
        class_name="Qwen2VLModel",
        params="2B",
        description="Qwen2-VL fine-tuned for OCR; not radiology-focused",
        colab_t4_native=True,
        benchmark_tier=BenchmarkTier.EXTRA,
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
        description="Qwen3-VL adapted for Basque/Galician/Catalan languages",
        colab_t4_native=False,
        benchmark_tier=BenchmarkTier.EXTRA,
        needs_4bit=True,
        default_kwargs={"model_name": "HiTZ/Latxa-Qwen3-VL-2B-Instruct", "load_in_4bit": True},
    ),
    
    # --- API Models ---
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
        benchmark_tier=BenchmarkTier.EXTRA,
        default_kwargs={},
    ),
    "gemini": ModelInfo(
        name="gemini",
        display_name="Gemini-1.5-Pro",
        category=ModelCategory.GENERALIST,
        cost=ModelCost.PAID,
        module_path="src.models.generalist.gemini",
        class_name="GeminiModel",
        params="unknown",
        description="Google's Gemini with vision (API)",
        colab_t4_native=True,  # API-based
        benchmark_tier=BenchmarkTier.EXTRA,
        default_kwargs={},
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_main_benchmark_models() -> List[str]:
    """Get all MAIN benchmark model names (13 models)."""
    return [m.name for m in MODEL_REGISTRY.values() if m.benchmark_tier == BenchmarkTier.MAIN]


def get_extra_track_models() -> List[str]:
    """Get all EXTRA track model names (8 models)."""
    return [m.name for m in MODEL_REGISTRY.values() if m.benchmark_tier == BenchmarkTier.EXTRA]


def get_models_by_benchmark_tier(tier: BenchmarkTier) -> List[ModelInfo]:
    """Get all ModelInfo objects for a specific benchmark tier."""
    return [m for m in MODEL_REGISTRY.values() if m.benchmark_tier == tier]


def get_models_by_category(category: ModelCategory, tier: Optional[BenchmarkTier] = None) -> List[ModelInfo]:
    """Get models by category, optionally filtered by tier."""
    result = [m for m in MODEL_REGISTRY.values() if m.category == category]
    if tier:
        result = [m for m in result if m.benchmark_tier == tier]
    return result


def get_free_models() -> List[ModelInfo]:
    """Get all free/open-source models."""
    return [m for m in MODEL_REGISTRY.values() if m.cost == ModelCost.FREE]


def get_paid_models() -> List[ModelInfo]:
    """Get all paid API models."""
    return [m for m in MODEL_REGISTRY.values() if m.cost == ModelCost.PAID]


def get_colab_compatible_models() -> List[ModelInfo]:
    """Get models that can run on Colab T4 (16GB VRAM)."""
    return [m for m in MODEL_REGISTRY.values() if m.colab_compatible]


def get_gated_access_models() -> List[ModelInfo]:
    """Get models that require gated HuggingFace access."""
    return [m for m in MODEL_REGISTRY.values() if m.gated_access]


def get_model_info(name: str) -> Optional[ModelInfo]:
    """Get ModelInfo by name."""
    return MODEL_REGISTRY.get(name)


def check_model_access(name: str) -> Dict[str, Any]:
    """Check if a model is accessible with helpful error message."""
    info = get_model_info(name)
    if not info:
        return {
            "accessible": False,
            "reason": "model_not_found",
            "message": f"Model '{name}' not in registry. Available main models: {get_main_benchmark_models()}",
        }
    
    # Check gated access
    if info.gated_access:
        return {
            "accessible": False,
            "reason": "gated_access",
            "message": f"Model '{name}' requires HuggingFace gated access approval.\n"
                      f"Steps:\n"
                      f"  1. Accept terms: https://huggingface.co/{info.default_kwargs.get('model_name', 'N/A')}\n"
                      f"  2. Get API token: https://huggingface.co/settings/tokens\n"
                      f"  3. Login: huggingface-cli login",
            "fix_url": f"https://huggingface.co/{info.default_kwargs.get('model_name', 'N/A')}",
        }
    
    # Check API key requirement for paid models
    if info.cost == ModelCost.PAID:
        api_key_var = "OPENAI_API_KEY" if "gpt" in name.lower() else "GOOGLE_API_KEY"
        import os
        if not os.environ.get(api_key_var):
            return {
                "accessible": False,
                "reason": "missing_api_key",
                "message": f"Model '{name}' requires {api_key_var} environment variable.\n"
                          f"Set: export {api_key_var}='your-key-here'",
            }
    
    return {"accessible": True, "reason": "ok", "message": f"Model '{name}' is accessible."}


def load_model(name: str):
    """Dynamically load and instantiate a model."""
    # Check access first
    access = check_model_access(name)
    if not access["accessible"]:
        extra_msg = ""
        if access["reason"] == BenchmarkTier.EXTRA.value:
            extra_msg = f"\nNote: This model is in EXTRA TRACK (out-of-scope). See GATING_REQUIREMENTS.md for setup."
        raise RuntimeError(f"{access['message']}{extra_msg}")
    
    info = get_model_info(name)
    if not info:
        raise ValueError(f"Unknown model: {name}")
    
    import importlib
    try:
        module = importlib.import_module(info.module_path)
        model_class = getattr(module, info.class_name)
        return model_class(**info.default_kwargs)
    except ModuleNotFoundError as e:
        raise ImportError(f"Cannot import {info.module_path}: {e}. Ensure model code is available.")
    except AttributeError as e:
        raise ImportError(f"Class {info.class_name} not found in {info.module_path}: {e}")


def print_model_table():
    """Print a formatted table of all models organized by tier and category."""
    print("\n" + "="*100)
    print("🎯 MAIN BENCHMARK MODELS (13)")
    print("="*100)
    print(f"{'Model':<25} {'Category':<15} {'Params':<8} {'T4 GPU':<12} {'Access':<15} Description")
    print("-"*100)
    
    for tier in [BenchmarkTier.MAIN, BenchmarkTier.EXTRA]:
        models = get_models_by_benchmark_tier(tier)
        if not models:
            continue
        
        if tier == BenchmarkTier.EXTRA:
            print("\n" + "="*100)
            print("📚 EXTRA TRACK MODELS (8 - Out of Scope)")
            print("="*100)
            print(f"{'Model':<25} {'Category':<15} {'Params':<8} {'T4 GPU':<12} {'Access':<15} Description")
            print("-"*100)
        
        for model in sorted(models, key=lambda m: (m.category.value, m.display_name)):
            # GPU compatibility
            if model.colab_t4_native:
                t4_status = "✅ Native"
            elif model.needs_4bit:
                t4_status = "✅ 4-bit"
            else:
                t4_status = "❌ No"
            
            # Access status
            if model.gated_access:
                access_status = "🔐 Gated"
            elif model.cost == ModelCost.PAID:
                access_status = "💳 API Key"
            else:
                access_status = "✅ Free"
            
            desc = model.description[:35] + ("..." if len(model.description) > 35 else "")
            print(f"{model.display_name:<25} {model.category.value:<15} {model.params:<8} {t4_status:<12} {access_status:<15} {desc}")
    
    print("\n" + "="*100 + "\n")


# Quick reference lists
MAIN_MODELS = get_main_benchmark_models()
EXTRA_MODELS = get_extra_track_models()

FREE_MODELS = [m.name for m in get_free_models()]
PAID_MODELS = [m.name for m in get_paid_models()]
COLAB_MODELS = [m.name for m in get_colab_compatible_models()]
GATED_MODELS = [m.name for m in get_gated_access_models()]

# Category-specific lists (main benchmark only)
GENERALIST_MODELS = [m.name for m in get_models_by_category(ModelCategory.GENERALIST, BenchmarkTier.MAIN)]
DOMAIN_ADAPTIVE_MODELS = [m.name for m in get_models_by_category(ModelCategory.DOMAIN_ADAPTIVE, BenchmarkTier.MAIN)]
SPECIALIST_MODELS = [m.name for m in get_models_by_category(ModelCategory.SPECIALIST, BenchmarkTier.MAIN)]
