"""
Phi-3 Vision Model Wrapper

Microsoft's small but powerful open-source vision-language model - completely free.
"""

import logging
from typing import Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class Phi3VisionModel(BaseRadiologyModel):
    """
    Phi-3 Vision model wrapper - Free, open-source from Microsoft.
    
    Small (4B params) but powerful - runs on consumer GPUs.
    Model: microsoft/Phi-3-vision-128k-instruct
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-vision-128k-instruct",
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
    
    def load(self) -> None:
        """Load Phi-3 Vision model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError("Please install: pip install transformers torch")
        
        logger.info(f"Loading {self.model_name}...")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
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
        prompt = prompt or "Generate a detailed radiology report for this chest X-ray. Include FINDINGS and IMPRESSION sections."
        
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"}
        ]
        
        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(prompt_text, [img], return_tensors="pt").to(self.device)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.processor.tokenizer.eos_token_id,
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
        
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{question}"}
        ]
        
        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(prompt_text, [img], return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=256)
        
        text = self.processor.decode(output[0], skip_special_tokens=True)
        return ModelOutput(text=text)
