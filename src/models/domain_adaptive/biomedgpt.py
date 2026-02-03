"""
BiomedCLIP-PubMedBERT Model Wrapper

Microsoft's biomedical vision-language model for medical imaging.
"""

import logging
from typing import Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class BiomedGPTModel(BaseRadiologyModel):
    """
    BiomedGPT model wrapper - Free, open-source biomedical VLM.
    
    Models available:
    - PharMolix/BioMedGPT-LM-7B
    - microsoft/BiomedCLIP-PubMedBERT
    """
    
    def __init__(
        self,
        model_name: str = "PharMolix/BioMedGPT-LM-7B",
        device: str = "cuda",
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.max_new_tokens = max_new_tokens
    
    def load(self) -> None:
        """Load BiomedGPT model."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("Please install: pip install transformers torch")
        
        logger.info(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        self._is_loaded = True
        logger.info(f"Loaded {self.model_name}")
    
    def generate_report(
        self,
        image: Union[Image.Image, str],
        prompt: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        """Generate radiology report."""
        if not self._is_loaded:
            self.load()
        
        prompt = prompt or "Generate a radiology report for this medical image."
        
        # BiomedGPT inference implementation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return ModelOutput(text=text)
    
    def answer_question(
        self,
        image: Union[Image.Image, str],
        question: str,
        **kwargs
    ) -> ModelOutput:
        """Answer a VQA question."""
        if not self._is_loaded:
            self.load()
        
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return ModelOutput(text=text)
