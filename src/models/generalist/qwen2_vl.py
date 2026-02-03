"""
Qwen2-VL Model Wrapper

Alibaba's open-source vision-language model - completely free to use.
"""

import logging
from typing import Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class Qwen2VLModel(BaseRadiologyModel):
    """
    Qwen2-VL model wrapper - Free, open-source from Alibaba.
    
    Models available:
    - Qwen/Qwen2-VL-7B-Instruct (7B params)
    - Qwen/Qwen2-VL-2B-Instruct (2B params, faster)
    - Qwen/Qwen2-VL-72B-Instruct (72B params, best quality)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
    
    def load(self) -> None:
        """Load Qwen2-VL model."""
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("Please install: pip install transformers torch qwen-vl-utils")
        
        logger.info(f"Loading {self.model_name}...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
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
        
        img = self.preprocess_image(image)
        prompt = prompt or "Generate a detailed radiology report for this chest X-ray."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return ModelOutput(text=output_text)
    
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
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        output_ids = self.model.generate(**inputs, max_new_tokens=256)
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return ModelOutput(text=output_text)
