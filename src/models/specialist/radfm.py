"""
RadFM Model Wrapper

Radiology Foundation Model - requires custom GitHub installation.
NOT available via standard HuggingFace transformers.
"""

import logging
from typing import Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class RadFMModel(BaseRadiologyModel):
    """
    RadFM (Radiology Foundation Model) - NOT SUPPORTED in this pipeline.
    
    RadFM requires custom installation from: https://github.com/chaoyi-wu/RadFM
    It is not available as a standard HuggingFace model.
    """
    
    def __init__(self, **kwargs):
        super().__init__(model_name="RadFM", model_type="local", **kwargs)
    
    def load(self) -> None:
        raise NotImplementedError(
            "RadFM requires custom GitHub installation from https://github.com/chaoyi-wu/RadFM. "
            "It cannot be loaded via HuggingFace transformers. "
            "Please remove 'radfm' from your experiment config."
        )
    
    def generate_report(self, image, prompt=None, **kwargs) -> ModelOutput:
        self.load()
    
    def answer_question(self, image, question, **kwargs) -> ModelOutput:
        self.load()
