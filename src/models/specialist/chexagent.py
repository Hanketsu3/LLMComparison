"""
CheXagent Model Wrapper

Stanford AIMI's CheXagent - a radiology foundation model.
"""

import logging
from typing import Dict, List, Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class CheXagentModel(BaseRadiologyModel):
    """CheXagent model wrapper for radiology tasks."""
    
    TASK_PROMPTS = {
        "report_generation": "Generate a detailed radiology report for this chest X-ray.",
        "view_classification": "What is the view of this chest X-ray? (PA/AP/Lateral)",
        "abnormality_detection": "List all abnormalities visible in this chest X-ray.",
        "phrase_grounding": "Locate the following finding in the image: {finding}",
    }
    
    def __init__(
        self,
        model_name: str = "StanfordAIMI/CheXagent-8b",
        device: str = "cuda",
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.max_new_tokens = max_new_tokens
    
    def load(self) -> None:
        """Load CheXagent model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError("Please install: pip install transformers torch")
        
        logger.info(f"Loading {self.model_name}...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
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
        prompt = prompt or self.TASK_PROMPTS["report_generation"]
        
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        
        text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return ModelOutput(text=text, metadata={"model": self.model_name})
    
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
        inputs = self.processor(images=img, text=question, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        
        text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return ModelOutput(text=text)
    
    def ground_finding(
        self,
        image: Union[Image.Image, str],
        finding: str,
        **kwargs
    ) -> ModelOutput:
        """Locate a finding in the image."""
        if not self._is_loaded:
            self.load()
        
        img = self.preprocess_image(image)
        prompt = self.TASK_PROMPTS["phrase_grounding"].format(finding=finding)
        
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        
        text = self.processor.decode(outputs[0], skip_special_tokens=True)
        bboxes = self._parse_bounding_boxes(text)
        
        return ModelOutput(text=text, bounding_boxes=bboxes)
    
    def _parse_bounding_boxes(self, text: str) -> List[Dict[str, float]]:
        """Parse bounding box coordinates from model output."""
        import re
        
        bboxes = []
        # Pattern: [x_min, y_min, x_max, y_max] or (x_min, y_min, x_max, y_max)
        pattern = r'[\[\(](\d+\.?\d*)[,\s]+(\d+\.?\d*)[,\s]+(\d+\.?\d*)[,\s]+(\d+\.?\d*)[\]\)]'
        
        for match in re.finditer(pattern, text):
            bboxes.append({
                "x_min": float(match.group(1)),
                "y_min": float(match.group(2)),
                "x_max": float(match.group(3)),
                "y_max": float(match.group(4)),
            })
        
        return bboxes
    
    @property
    def supports_grounding(self) -> bool:
        return True
