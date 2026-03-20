"""
Phi-3.5 Vision Model Wrapper

Microsoft's small but powerful open-source vision-language model - completely free.
Updated to Phi-3.5-vision-instruct for transformers >= 4.45 compatibility.
"""

import logging
from typing import Optional, Union
from PIL import Image
from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class Phi3VisionModel(BaseRadiologyModel):
    """
    Phi-3.5 Vision model wrapper - Free, open-source from Microsoft.
    
    Uses Phi-3.5-vision-instruct (4.2B params) instead of older Phi-3-vision
    which has DynamicCache incompatibility with transformers >= 4.45.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-vision-instruct",
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
    
    def load(self) -> None:
        """Load Phi-3.5 Vision model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise ImportError("Please install: pip install transformers torch")
        
        logger.info(f"Loading {self.model_name}...")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            num_crops=4,  # Phi-3.5 supports multi-crop
        )
        
        # Phi-3.5 Vision hotfix for transformers >= 4.45 (from_legacy_cache was removed)
        import transformers
        if hasattr(transformers, "cache_utils") and hasattr(transformers.cache_utils, "DynamicCache"):
            if not hasattr(transformers.cache_utils.DynamicCache, "from_legacy_cache"):
                @classmethod
                def from_legacy_cache(cls, past_key_values, max_cache_len=None):
                    cache = cls()
                    if past_key_values is not None:
                        for layer_idx in range(len(past_key_values)):
                            key_states, value_states = past_key_values[layer_idx]
                            cache.update(key_states, value_states, layer_idx)
                    return cache
                transformers.cache_utils.DynamicCache.from_legacy_cache = from_legacy_cache
            
            if not hasattr(transformers.cache_utils.DynamicCache, "get_max_length"):
                def get_max_length(self):
                    return None
                transformers.cache_utils.DynamicCache.get_max_length = get_max_length
        
        
        # Try flash_attention_2 first, fall back to eager (Colab compatibility)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                _attn_implementation="flash_attention_2",
            )
        except Exception:
            logger.info("flash_attention_2 not available, using eager attention")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                _attn_implementation="eager",
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
        
        # Phi-3.5 uses placeholder tokens for images
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
        """Answer a VQA question."""
        if not self._is_loaded:
            self.load()
        
        img = self.preprocess_image(image)
        vqa_prompt = self.format_vqa_prompt(question)
        
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{vqa_prompt}"}
        ]
        
        prompt_text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(prompt_text, [img], return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=50)
        
        # Only decode the NEW tokens (trim input tokens)
        generated_ids = output[:, inputs['input_ids'].shape[1]:]
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return ModelOutput(text=text)
