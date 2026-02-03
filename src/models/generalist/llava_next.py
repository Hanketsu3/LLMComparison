"""
LLaVA-NeXT Model Wrapper

Open-source successor to LLaVA - completely free to use.
"""

import logging
from typing import Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class LLaVANextModel(BaseRadiologyModel):
    """
    LLaVA-NeXT (LLaVA 1.6) model wrapper - Free, open-source.
    
    Models available:
    - llava-hf/llava-v1.6-mistral-7b-hf
    - llava-hf/llava-v1.6-vicuna-7b-hf
    - llava-hf/llava-v1.6-vicuna-13b-hf
    - llava-hf/llava-v1.6-34b-hf
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
    
    def load(self) -> None:
        """Load LLaVA-NeXT model."""
        try:
            import torch
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
        except ImportError:
            raise ImportError("Please install: pip install transformers torch")
        
        logger.info(f"Loading {self.model_name}...")
        
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        
        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
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
        prompt = prompt or "[INST] <image>\nGenerate a detailed radiology report for this chest X-ray with FINDINGS and IMPRESSION sections. [/INST]"
        
        inputs = self.processor(prompt, img, return_tensors="pt").to(self.device)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        
        text = self.processor.decode(output[0], skip_special_tokens=True)
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
        prompt = f"[INST] <image>\n{question} [/INST]"
        
        inputs = self.processor(prompt, img, return_tensors="pt").to(self.device)
        
        output = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(output[0], skip_special_tokens=True)
        
        return ModelOutput(text=text)
