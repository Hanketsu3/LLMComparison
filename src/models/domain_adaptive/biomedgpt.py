"""
BiomedGPT Model Wrapper

BiomedGPT - NOT SUPPORTED for multimodal inference in this pipeline.
The HuggingFace version does not support image+text multimodal input.
"""

import logging
from typing import Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class BiomedGPTModel(BaseRadiologyModel):
    """
    BiomedGPT model - NOT SUPPORTED for multimodal tasks.
    
    The available HuggingFace versions (PharMolix/BioMedGPT-LM-7B) are 
    text-only language models and do NOT accept image inputs.
    """
    
    def __init__(self, **kwargs):
        super().__init__(model_name="BiomedGPT", model_type="local", **kwargs)
    
    def load(self) -> None:
        raise NotImplementedError(
            "BiomedGPT (PharMolix/BioMedGPT-LM-7B) is a text-only model and does NOT support "
            "image inputs. It cannot be used for multimodal radiology tasks. "
            "Please remove 'biomedgpt' from your experiment config."
        )
    
    def generate_report(self, image, prompt=None, **kwargs) -> ModelOutput:
        self.load()
    
    def answer_question(self, image, question, **kwargs) -> ModelOutput:
        self.load()
