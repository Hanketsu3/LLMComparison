"""Utility modules."""

# Lazy imports to avoid dependency issues when packages not installed

def __getattr__(name):
    """Lazy import for utils submodules."""
    if name == "setup_logging":
        from src.utils.logging import setup_logging
        return setup_logging
    elif name == "StatisticalTester":
        from src.utils.statistical_tests import StatisticalTester
        return StatisticalTester
    elif name in ("PromptManager", "PromptTemplate"):
        from src.utils.prompt_manager import PromptManager, PromptTemplate
        return PromptManager if name == "PromptManager" else PromptTemplate
    elif name in ("MODEL_REGISTRY", "ModelInfo", "ModelCost", "ModelCategory",
                  "get_free_models", "get_paid_models", "get_colab_compatible_models",
                  "load_model", "print_model_table", "FREE_MODELS", "PAID_MODELS", "COLAB_MODELS"):
        from src.utils import model_registry
        return getattr(model_registry, name)
    raise AttributeError(f"module 'src.utils' has no attribute '{name}'")


__all__ = [
    "setup_logging", 
    "StatisticalTester", 
    "PromptManager", 
    "PromptTemplate",
    "MODEL_REGISTRY",
    "ModelInfo",
    "ModelCost",
    "ModelCategory",
    "get_free_models",
    "get_paid_models",
    "get_colab_compatible_models",
    "load_model",
    "print_model_table",
    "FREE_MODELS",
    "PAID_MODELS",
    "COLAB_MODELS",
]
