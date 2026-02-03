"""
GPT-4V (Vision) Model Wrapper

OpenAI's GPT-4 with vision capabilities for radiology tasks.
"""

import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class GPT4VModel(BaseRadiologyModel):
    """GPT-4V model wrapper for radiology tasks."""
    
    DEFAULT_PROMPTS = {
        "report_generation": """You are an expert radiologist. Analyze the provided 
chest X-ray and generate a detailed radiology report.

Your report should include:
FINDINGS: Detailed observations
IMPRESSION: Clinical interpretation""",
        
        "vqa": "You are an expert radiologist. Answer the question accurately.",
    }
    
    def __init__(
        self,
        model_name: str = "gpt-4-vision-preview",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        few_shot_examples: Optional[List[Dict]] = None,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="api", **kwargs)
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.few_shot_examples = few_shot_examples or []
        self.client = None
    
    def load(self) -> None:
        """Initialize OpenAI client."""
        import os
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        
        self.client = OpenAI(api_key=api_key)
        self._is_loaded = True
        logger.info(f"Loaded {self.model_name}")
    
    def _encode_image(self, image: Union[Image.Image, str]) -> str:
        """Encode image to base64."""
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.standard_b64encode(f.read()).decode("utf-8")
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    
    def _build_messages(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict]:
        """Build message list for API call."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add few-shot examples
        for example in self.few_shot_examples:
            if "image" in example:
                img_b64 = self._encode_image(example["image"])
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": example.get("prompt", "Analyze this image.")}
                    ]
                })
            if "response" in example:
                messages.append({"role": "assistant", "content": example["response"]})
        
        # Add main query
        img_b64 = self._encode_image(image)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt}
            ]
        })
        
        return messages
    
    def generate_report(
        self,
        image: Union[Image.Image, str],
        prompt: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        """Generate radiology report."""
        if not self._is_loaded:
            self.load()
        
        prompt = prompt or "Generate a detailed radiology report for this chest X-ray."
        system = self.DEFAULT_PROMPTS["report_generation"]
        
        messages = self._build_messages(image, prompt, system)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        
        text = response.choices[0].message.content
        findings, impression = self._parse_report(text)
        
        return ModelOutput(
            text=text,
            findings=findings,
            impression=impression,
            metadata={"model": self.model_name, "usage": dict(response.usage)}
        )
    
    def answer_question(
        self,
        image: Union[Image.Image, str],
        question: str,
        **kwargs
    ) -> ModelOutput:
        """Answer a VQA question."""
        if not self._is_loaded:
            self.load()
        
        messages = self._build_messages(
            image, 
            f"Question: {question}\nAnswer:",
            self.DEFAULT_PROMPTS["vqa"]
        )
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=256,
            temperature=0.0,
        )
        
        return ModelOutput(
            text=response.choices[0].message.content,
            metadata={"model": self.model_name}
        )
    
    def _parse_report(self, text: str) -> tuple:
        """Extract findings and impression from report text."""
        import re
        
        findings = ""
        impression = ""
        
        findings_match = re.search(r"FINDINGS[:\s]*(.*?)(?=IMPRESSION|$)", text, re.I | re.S)
        if findings_match:
            findings = findings_match.group(1).strip()
        
        impression_match = re.search(r"IMPRESSION[:\s]*(.*?)$", text, re.I | re.S)
        if impression_match:
            impression = impression_match.group(1).strip()
        
        return findings, impression
