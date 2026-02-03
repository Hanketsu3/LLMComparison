"""
MiniCPM-V Model Wrapper

Open-source multimodal LLM from OpenBMB - completely free to use.
"""

import logging
from typing import Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class MiniCPMVModel(BaseRadiologyModel):
    """
    MiniCPM-V model wrapper - Free, open-source multimodal LLM.
    
    Models available:
    - openbmb/MiniCPM-V-2_6 (8B params, best quality)
    - openbmb/MiniCPM-Llama3-V-2_5 (8B params)
    - openbmb/MiniCPM-V-2 (3B params, faster)
    """
    
    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-V-2_6",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
    
    def load(self) -> None:
        """Load MiniCPM-V model."""
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
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        
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
        prompt = prompt or "Generate a detailed radiology report for this chest X-ray with FINDINGS and IMPRESSION sections."
        
        msgs = [{"role": "user", "content": [img, prompt]}]
        
        answer = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
        )
        
        return ModelOutput(text=answer)
    
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
        msgs = [{"role": "user", "content": [img, question]}]
        
        answer = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
        )
        
        return ModelOutput(text=answer)
