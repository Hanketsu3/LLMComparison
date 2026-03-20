"""
Backend Interface for Model Inference

Provides abstraction between different inference backends (transformers, vLLM, etc.)
to allow flexible swapping without changing model code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from PIL import Image


class BackendType(Enum):
    """Supported inference backends."""
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


@dataclass
class GenerationConfig:
    """Unified generation configuration across backends."""
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    num_beams: int = 1
    do_sample: bool = False
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    
    # Backend-specific
    use_kv_cache: bool = True
    use_flash_attention: bool = False
    use_flash_attention_2: bool = False
    attention_sink_window: Optional[int] = None  # For T4
    
    # Batching (for vLLM)
    enable_batching: bool = False
    batch_size: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, filtering out None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class InferenceResult:
    """Unified result format from all backends."""
    text: str
    tokens: int
    latency_ms: float
    memory_used_mb: Optional[float] = None
    backend: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BackendInterface(ABC):
    """Abstract base class for inference backends."""
    
    def __init__(
        self,
        model_name: str,
        backend_type: BackendType,
        device: str = "cuda",
        dtype: str = "auto",
        **kwargs
    ):
        self.model_name = model_name
        self.backend_type = backend_type
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs
        self._is_loaded = False
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        pass
    
    @abstractmethod
    def infer(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        config: GenerationConfig,
    ) -> InferenceResult:
        """Run single inference."""
        pass
    
    @abstractmethod
    def infer_batch(
        self,
        images: List[Union[Image.Image, str]],
        prompts: List[str],
        config: GenerationConfig,
    ) -> List[InferenceResult]:
        """Run batch inference (if supported)."""
        pass
    
    def supports_batching(self) -> bool:
        """Check if backend supports batching."""
        return False
    
    def estimate_memory_mb(self, model_size_b: float) -> float:
        """Estimate memory in MB for this model."""
        # Rough: model size + overhead
        return model_size_b * 1.3
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, backend={self.backend_type.value})"


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.model_registry import ModelInfo


class BackendRegistry:
    """Registry for matching models to backends."""
    
    _backend_map: Dict[str, BackendType] = {
        # Transformers backends (default)
        "qwen2-vl-2b": BackendType.TRANSFORMERS,
        "qwen2.5-vl-3b": BackendType.TRANSFORMERS,
        "qwen3-vl-2b": BackendType.TRANSFORMERS,
        "phi3-vision": BackendType.TRANSFORMERS,
        "smolvlm2-2.2b": BackendType.TRANSFORMERS,
        "internvl2-2b": BackendType.TRANSFORMERS,
        "internvl2-4b": BackendType.TRANSFORMERS,
        "llama3-vision": BackendType.TRANSFORMERS,
        "llava-med": BackendType.TRANSFORMERS,
        "chexagent": BackendType.TRANSFORMERS,
        "llava-rad": BackendType.TRANSFORMERS,
        "radfm": BackendType.TRANSFORMERS,
        "medgemma-4b": BackendType.TRANSFORMERS,
        "biomedgpt": BackendType.TRANSFORMERS,
        
        # vLLM candidates (optional, if installed)
        # Uncomment when vLLM is available
        # "llama3-vision": BackendType.VLLM,
        # "chexagent": BackendType.VLLM,
    }
    
    @classmethod
    def get_backend_for_model(cls, model_name: str) -> BackendType:
        """Get recommended backend for a model."""
        return cls._backend_map.get(model_name, BackendType.TRANSFORMERS)
    
    @classmethod
    def register_model_backend(cls, model_name: str, backend: BackendType) -> None:
        """Register custom backend for a model."""
        cls._backend_map[model_name] = backend
