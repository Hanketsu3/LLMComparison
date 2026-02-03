"""
Gemini Pro Vision Model Wrapper

Google's Gemini 1.5 Pro for radiology tasks.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class GeminiModel(BaseRadiologyModel):
    """Gemini Pro Vision model wrapper."""
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="api", **kwargs)
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = None
    
    def load(self) -> None:
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install: pip install google-generativeai")
        
        api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not provided")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
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
        prompt = prompt or """You are an expert radiologist. Analyze this chest X-ray.

Generate a report with:
FINDINGS: Detailed observations
IMPRESSION: Clinical interpretation"""
        
        response = self.model.generate_content(
            [prompt, img],
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        )
        
        text = response.text
        findings, impression = self._parse_report(text)
        
        return ModelOutput(text=text, findings=findings, impression=impression)
    
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
        prompt = f"You are an expert radiologist. Question: {question}\nAnswer:"
        
        response = self.model.generate_content([prompt, img])
        return ModelOutput(text=response.text)
    
    def _parse_report(self, text: str) -> tuple:
        import re
        findings = ""
        impression = ""
        
        m = re.search(r"FINDINGS[:\s]*(.*?)(?=IMPRESSION|$)", text, re.I | re.S)
        if m:
            findings = m.group(1).strip()
        
        m = re.search(r"IMPRESSION[:\s]*(.*?)$", text, re.I | re.S)
        if m:
            impression = m.group(1).strip()
        
        return findings, impression
