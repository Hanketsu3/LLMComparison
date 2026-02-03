"""
Base Model class for all radiology LLMs.

Provides a unified interface for generalist, domain-adaptive, and specialist models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class ModelOutput:
    """Standard output format for all models."""
    text: str
    findings: Optional[str] = None
    impression: Optional[str] = None
    bounding_boxes: Optional[List[Dict[str, float]]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseRadiologyModel(ABC):
    """Abstract base class for all radiology models."""
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "local",  # "local" or "api"
        device: str = "cuda",
        **kwargs
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.model = None
        self.processor = None
        self._is_loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def generate_report(
        self,
        image: Union[Image.Image, str],
        prompt: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        """Generate a radiology report for the given image."""
        pass
    
    @abstractmethod
    def answer_question(
        self,
        image: Union[Image.Image, str],
        question: str,
        **kwargs
    ) -> ModelOutput:
        """Answer a question about the given image."""
        pass
    
    def ground_finding(
        self,
        image: Union[Image.Image, str],
        finding: str,
        **kwargs
    ) -> ModelOutput:
        """Locate a finding in the image (if supported)."""
        raise NotImplementedError(f"{self.model_name} does not support grounding")
    
    def preprocess_image(self, image: Union["Image.Image", str]) -> "Image.Image":
        """Preprocess an image for the model."""
        from PIL import Image
        
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        return image
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @property
    def supports_grounding(self) -> bool:
        return False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
