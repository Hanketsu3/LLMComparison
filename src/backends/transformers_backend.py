"""
Transformers Backend Implementation

Wrapper around HuggingFace transformers for unified inference interface.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from src.backends.base import BackendInterface, BackendType, GenerationConfig, InferenceResult

logger = logging.getLogger(__name__)


class TransformersBackend(BackendInterface):
    """HuggingFace Transformers inference backend."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "auto",
        quantization_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            backend_type=BackendType.TRANSFORMERS,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.quantization_config = quantization_config or {}
    
    def load(self) -> None:
        """Load model using transformers."""
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        logger.info(f"Loading {self.model_name} via Transformers...")
        
        # Determine dtype
        if self.dtype == "auto":
            dtype_map = {"cuda": torch.bfloat16, "cpu": torch.float32}
            torch_dtype = dtype_map.get(self.device, torch.float32)
        elif self.dtype == "fp16":
            torch_dtype = torch.float16
        elif self.dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load processor: {e}, trying without trust_remote_code")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Load model with quantization if specified
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        # Apply quantization if requested
        if self.quantization_config.get("load_in_4bit"):
            try:
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["quantization_config"] = bnb_config
                logger.info("Using 4-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not available, skipping 4-bit quantization")
        
        elif self.quantization_config.get("load_in_8bit"):
            try:
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["quantization_config"] = bnb_config
                logger.info("Using 8-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not available, skipping 8-bit quantization")
        
        # Load model
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                **load_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to load model with AutoModel: {e}")
            raise
        
        self._is_loaded = True
        logger.info(f"Loaded {self.model_name}")
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        self._is_loaded = False
        self.model = None
        self.processor = None
    
    def infer(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        config: GenerationConfig,
    ) -> InferenceResult:
        """Run single inference with transformers."""
        if not self._is_loaded:
            self.load()
        
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image)
        
        start_time = time.time()
        
        try:
            # Preprocess
            inputs = self._prepare_inputs(image, prompt, config)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    num_beams=config.num_beams,
                    use_cache=config.use_kv_cache,
                )
            
            # Decode
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            latency_ms = (time.time() - start_time) * 1000
            tokens = len(generated_ids[0])
            
            return InferenceResult(
                text=text,
                tokens=tokens,
                latency_ms=latency_ms,
                backend="transformers",
                metadata={
                    "model_name": self.model_name,
                    "quantized": bool(self.quantization_config),
                }
            )
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise
    
    def infer_batch(
        self,
        images: List[Union[Image.Image, str]],
        prompts: List[str],
        config: GenerationConfig,
    ) -> List[InferenceResult]:
        """Run batch inference (limited support)."""
        # Most transformers models don't support dynamic batching
        # So we run iteratively
        return [
            self.infer(image, prompt, config)
            for image, prompt in zip(images, prompts)
        ]
    
    def supports_batching(self) -> bool:
        """Transformers usually requires fixed batch size."""
        return False
    
    def _prepare_inputs(
        self,
        image: Image.Image,
        prompt: str,
        config: GenerationConfig,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for model.generate().
        
        Subclasses override this for model-specific preprocessing.
        """
        # Default: processor handles image + text
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        return inputs
