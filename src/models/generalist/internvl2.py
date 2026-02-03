"""
InternVL2 Model Wrapper

OpenGVLab's open-source vision-language model - completely free to use.
"""

import logging
from typing import Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class InternVL2Model(BaseRadiologyModel):
    """
    InternVL2 model wrapper - Free, open-source from OpenGVLab.
    
    Models available:
    - OpenGVLab/InternVL2-8B (8B params)
    - OpenGVLab/InternVL2-4B (4B params)
    - OpenGVLab/InternVL2-2B (2B params, fastest)
    - OpenGVLab/InternVL2-26B (26B params)
    """
    
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL2-8B",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
    
    def load(self) -> None:
        """Load InternVL2 model."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("Please install: pip install transformers torch")
        
        logger.info(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        self.model.eval()
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
        prompt = prompt or "<image>\nGenerate a detailed radiology report for this chest X-ray with FINDINGS and IMPRESSION sections."
        
        response = self.model.chat(
            self.tokenizer,
            img,
            prompt,
            generation_config={"max_new_tokens": self.max_new_tokens}
        )
        
        return ModelOutput(text=response)
    
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
        prompt = f"<image>\n{question}"
        
        response = self.model.chat(
            self.tokenizer,
            img,
            prompt,
            generation_config={"max_new_tokens": 256}
        )
        
        return ModelOutput(text=response)
