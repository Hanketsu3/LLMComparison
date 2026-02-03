"""
LLaVA-Med Model Wrapper

Microsoft's LLaVA adapted for biomedical domain.
"""

import logging
from typing import Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class LLaVAMedModel(BaseRadiologyModel):
    """LLaVA-Med model wrapper for biomedical VQA and report generation."""
    
    def __init__(
        self,
        model_name: str = "microsoft/llava-med-v1.5-mistral-7b",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
    
    def load(self) -> None:
        """Load LLaVA-Med model."""
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        except ImportError:
            raise ImportError("Please install: pip install transformers torch")
        
        logger.info(f"Loading {self.model_name}...")
        
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
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
        """Generate radiology report."""
        if not self._is_loaded:
            self.load()
        
        img = self.preprocess_image(image)
        prompt = prompt or "<image>\nGenerate a detailed radiology report for this chest X-ray."
        
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        
        text = self.processor.decode(outputs[0], skip_special_tokens=True)
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
        
        img = self.preprocess_image(image)
        prompt = f"<image>\nQuestion: {question}\nAnswer:"
        
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        
        text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return ModelOutput(text=text)
