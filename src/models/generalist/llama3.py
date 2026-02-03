"""
Llama-3 Vision Model Wrapper

Meta's Llama-3 for radiology tasks (local deployment).
"""

import logging
from typing import Any, Dict, Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class Llama3Model(BaseRadiologyModel):
    """Llama-3 model wrapper for local deployment."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
    
    def load(self) -> None:
        """Load the model using transformers."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError("Please install: pip install transformers torch")
        
        logger.info(f"Loading {self.model_name}...")
        
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        self._is_loaded = True
        logger.info(f"Loaded {self.model_name}")
    
    def generate_report(
        self,
        image: Union[Image.Image, str],
        prompt: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        """Generate radiology report (text-only for base Llama-3)."""
        if not self._is_loaded:
            self.load()
        
        # Note: Base Llama-3 doesn't support images directly
        # This would need LLaVA or similar multimodal adapter
        prompt = prompt or "Generate a radiology report based on the image description."
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        
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
        
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return ModelOutput(text=text)
