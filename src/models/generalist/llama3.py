"""
Llama-3.2 Vision Model Wrapper

Meta's Llama-3.2 Vision for radiology tasks (multimodal, local deployment).
"""

import logging
from typing import Any, Dict, Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class Llama3Model(BaseRadiologyModel):
    """
    Llama-3.2 Vision model wrapper for local deployment.
    
    Models available:
    - meta-llama/Llama-3.2-11B-Vision-Instruct (11B, multimodal)
    - meta-llama/Llama-3.2-90B-Vision-Instruct (90B, multimodal)
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
    
    def load(self) -> None:
        """Load the Llama-3.2 Vision model."""
        try:
            import torch
            from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
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
        self.model = MllamaForConditionalGeneration.from_pretrained(
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
        """Generate radiology report from image."""
        if not self._is_loaded:
            self.load()
        
        img = self.preprocess_image(image)
        prompt = prompt or "Generate a detailed radiology report for this chest X-ray. Include FINDINGS and IMPRESSION sections."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=img, text=input_text, return_tensors="pt").to(self.model.device)
        
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        # Only decode the NEW tokens (trim input tokens)
        generated_ids = output[:, inputs['input_ids'].shape[1]:]
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return ModelOutput(text=text)
    
    def answer_question(
        self,
        image: Union[Image.Image, str],
        question: str,
        **kwargs
    ) -> ModelOutput:
        """Answer a VQA question about the image."""
        if not self._is_loaded:
            self.load()
        
        img = self.preprocess_image(image)
        vqa_prompt = self.format_vqa_prompt(question)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": vqa_prompt},
                ],
            }
        ]
        
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=img, text=input_text, return_tensors="pt").to(self.model.device)
        
        output = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
        # Only decode the NEW tokens (trim input tokens)
        generated_ids = output[:, inputs['input_ids'].shape[1]:]
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return ModelOutput(text=text)
