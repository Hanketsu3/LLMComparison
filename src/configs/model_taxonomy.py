"""
Model Taxonomy - Radiology Benchmark vs Extra Track

Defines which models are in the main radiology comparison benchmark
and which are experimental/out-of-scope helpers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class BenchmarkTier(Enum):
    """Defines benchmark tier for a model."""
    MAIN = "main"              # Core radiology comparison
    EXTRA = "extra"            # Out-of-scope experimental track


class ModelCategory(Enum):
    """Resource categories within main benchmark."""
    GENERALIST = "generalist"           # General-purpose VLMs
    DOMAIN_ADAPTIVE = "domain_adaptive" # Medical/biomedical adapted
    SPECIALIST = "specialist"           # Radiology-specific


@dataclass
class ModelTaxonomy:
    """Single model's taxonomy classification."""
    model_name: str
    display_name: str
    tier: BenchmarkTier
    category: ModelCategory
    reason: str  # Why it's in this tier/category
    params: str  # "2B", "7B", "8B", etc.


# ============================================================================
# MAIN RADIOLOGY BENCHMARK
# ============================================================================

MAIN_BENCHMARK_MODELS: Dict[str, ModelTaxonomy] = {
    # --- Generalist (lightweight general-purpose) ---
    "qwen2-vl-2b": ModelTaxonomy(
        model_name="qwen2-vl-2b",
        display_name="Qwen2-VL-2B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.GENERALIST,
        reason="Lightweight general VLM, Colab T4 native, reasonable radiology performance",
        params="2B",
    ),
    "qwen2.5-vl-3b": ModelTaxonomy(
        model_name="qwen2.5-vl-3b",
        display_name="Qwen2.5-VL-3B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.GENERALIST,
        reason="Improved Qwen2-VL with better instruction following",
        params="3B",
    ),
    "qwen3-vl-2b": ModelTaxonomy(
        model_name="qwen3-vl-2b",
        display_name="Qwen3-VL-2B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.GENERALIST,
        reason="Latest Qwen series, compact multimodal reasoning",
        params="2B",
    ),
    "phi3-vision": ModelTaxonomy(
        model_name="phi3-vision",
        display_name="Phi-3.5-Vision",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.GENERALIST,
        reason="Microsoft compact VLM, good quality-to-size ratio",
        params="4.2B",
    ),
    "smolvlm2-2.2b": ModelTaxonomy(
        model_name="smolvlm2-2.2b",
        display_name="SmolVLM2-2.2B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.GENERALIST,
        reason="Hugging Face lightweight model optimized for resource-constrained settings",
        params="2.2B",
    ),
    "internvl2-2b": ModelTaxonomy(
        model_name="internvl2-2b",
        display_name="InternVL2-2B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.GENERALIST,
        reason="Efficient vision-language model, high performance-to-parameter ratio",
        params="2B",
    ),
    "internvl2-4b": ModelTaxonomy(
        model_name="internvl2-4b",
        display_name="InternVL2-4B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.GENERALIST,
        reason="Balanced InternVL2 variant for better quality on 24GB+ GPUs",
        params="4B",
    ),
    "llama3-vision": ModelTaxonomy(
        model_name="llama3-vision",
        display_name="Llama-3.2-Vision-11B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.GENERALIST,
        reason="Meta's multimodal Llama; requires 4-bit for T4, gated access",
        params="11B",
    ),
    
    # --- Domain-Adaptive (medical/biomedical specialized) ---
    "llava-med": ModelTaxonomy(
        model_name="llava-med",
        display_name="LLaVA-Med-7B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.DOMAIN_ADAPTIVE,
        reason="Vision-language model fine-tuned on medical images and clinical text",
        params="7B",
    ),
    "medgemma-4b": ModelTaxonomy(
        model_name="medgemma-4b",
        display_name="MedGemma-4B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.DOMAIN_ADAPTIVE,
        reason="Google's medical domain-adapted Gemma variant",
        params="4B",
    ),
    "biomedgpt": ModelTaxonomy(
        model_name="biomedgpt",
        display_name="BiomedGPT",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.DOMAIN_ADAPTIVE,
        reason="Medical text corpus pre-trained variant",
        params="7B",
    ),
    
    # --- Specialist (radiology-specific) ---
    "chexagent": ModelTaxonomy(
        model_name="chexagent",
        display_name="CheXagent-8B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.SPECIALIST,
        reason="Stanford AIMI chest X-ray expert; phrase grounding support",
        params="8B",
    ),
    "llava-rad": ModelTaxonomy(
        model_name="llava-rad",
        display_name="LLaVA-Rad-8B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.SPECIALIST,
        reason="LLaVA fine-tuned on radiology reports and images",
        params="8B",
    ),
    "radfm": ModelTaxonomy(
        model_name="radfm",
        display_name="RadFM-7B",
        tier=BenchmarkTier.MAIN,
        category=ModelCategory.SPECIALIST,
        reason="Foundation model trained on radiology images and reports",
        params="7B",
    ),
}

# ============================================================================
# EXTRA TRACK (Out-of-Scope Helpers)
# ============================================================================

EXTRA_TRACK_MODELS: Dict[str, ModelTaxonomy] = {
    # --- OCR / Document / Chart Analysis (not radiology) ---
    "got-ocr2": ModelTaxonomy(
        model_name="got-ocr2",
        display_name="GOT-OCR-2.0",
        tier=BenchmarkTier.EXTRA,
        category=ModelCategory.GENERALIST,
        reason="OCR specialist for documents, formulas, scene text. Out-of-scope for radiology.",
        params="0.58B",
    ),
    "nougat-base": ModelTaxonomy(
        model_name="nougat-base",
        display_name="Nougat-Base",
        tier=BenchmarkTier.EXTRA,
        category=ModelCategory.GENERALIST,
        reason="Document/academic paper parsing. Out-of-scope for radiology.",
        params="0.35B",
    ),
    "matcha-chartqa": ModelTaxonomy(
        model_name="matcha-chartqa",
        display_name="MatCha-ChartQA",
        tier=BenchmarkTier.EXTRA,
        category=ModelCategory.GENERALIST,
        reason="Chart/graph QA model. Out-of-scope for radiology.",
        params="0.67B",
    ),
    "qwen2-vl-ocr": ModelTaxonomy(
        model_name="qwen2-vl-ocr",
        display_name="Qwen2-VL-OCR-2B",
        tier=BenchmarkTier.EXTRA,
        category=ModelCategory.DOMAIN_ADAPTIVE,
        reason="Qwen2-VL fine-tuned for OCR. Not radiology-focused.",
        params="2B",
    ),
    
    # --- Language-Adaptation (specific language only) ---
    "latxa-qwen3-vl-2b": ModelTaxonomy(
        model_name="latxa-qwen3-vl-2b",
        display_name="Latxa-Qwen3-VL-2B",
        tier=BenchmarkTier.EXTRA,
        category=ModelCategory.DOMAIN_ADAPTIVE,
        reason="Qwen3-VL adapted for Basque/Galician/Catalan. Not English radiology benchmark.",
        params="2B",
    ),
    
    # --- API Models (cost tracking required) ---
    "gpt4v": ModelTaxonomy(
        model_name="gpt4v",
        display_name="GPT-4 Vision",
        tier=BenchmarkTier.EXTRA,
        category=ModelCategory.GENERALIST,
        reason="Proprietary API; cost tracking required; optional for comparison.",
        params="unknown",
    ),
    "gemini": ModelTaxonomy(
        model_name="gemini",
        display_name="Gemini-1.5-Pro",
        tier=BenchmarkTier.EXTRA,
        category=ModelCategory.GENERALIST,
        reason="Proprietary API; cost tracking required; optional for comparison.",
        params="unknown",
    ),
}

# ============================================================================
# CONSOLIDATED REGISTRY
# ============================================================================

ALL_MODELS = {**MAIN_BENCHMARK_MODELS, **EXTRA_TRACK_MODELS}


def get_model_taxonomy(model_name: str) -> ModelTaxonomy:
    """Retrieve taxonomy info for a model."""
    if model_name not in ALL_MODELS:
        raise ValueError(f"Model '{model_name}' not found in taxonomy. Available: {list(ALL_MODELS.keys())}")
    return ALL_MODELS[model_name]


def get_models_by_tier(tier: BenchmarkTier) -> Dict[str, ModelTaxonomy]:
    """Get all models in a specific tier."""
    return {name: info for name, info in ALL_MODELS.items() if info.tier == tier}


def get_models_by_category(category: ModelCategory, tier: BenchmarkTier = BenchmarkTier.MAIN) -> Dict[str, ModelTaxonomy]:
    """Get all models in a specific category within a tier."""
    return {
        name: info for name, info in ALL_MODELS.items()
        if info.category == category and info.tier == tier
    }


def validate_model_list(model_names: List[str]) -> List[str]:
    """Validate a list of model names and return only valid ones with warnings."""
    valid = []
    for name in model_names:
        if name not in ALL_MODELS:
            print(f"⚠️  Warning: Model '{name}' not found in taxonomy, skipping.")
        else:
            valid.append(name)
    return valid


def print_taxonomy_summary() -> None:
    """Print a summary of the taxonomy."""
    print("\n" + "=" * 80)
    print("📊 RADIOLOGY BENCHMARK MODEL TAXONOMY")
    print("=" * 80)
    
    for tier in BenchmarkTier:
        models = get_models_by_tier(tier)
        if not models:
            continue
        
        tier_title = "🎯 MAIN BENCHMARK" if tier == BenchmarkTier.MAIN else "📚 EXTRA TRACK"
        print(f"\n{tier_title}:")
        print("-" * 80)
        
        for category in ModelCategory:
            category_models = {
                k: v for k, v in models.items() if v.category == category
            }
            if not category_models:
                continue
            
            print(f"\n  {category.value.upper()}:")
            for name, info in sorted(category_models.items()):
                print(f"    • {info.display_name:30s} ({info.params:>5s}) - {info.reason}")
    
    print("\n" + "=" * 80 + "\n")
