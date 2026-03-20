"""
Generic Vision Chat Model Wrapper

A flexible wrapper for modern image-text chat models that support
AutoProcessor + generation APIs through Transformers.
"""

import logging
from typing import Optional, Union

from PIL import Image

from src.models.base_model import BaseRadiologyModel, ModelOutput

logger = logging.getLogger(__name__)


class GenericVisionChatModel(BaseRadiologyModel):
    """Generic wrapper for VLMs with chat-style image+text prompting."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_4bit: bool = False,
        max_new_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(model_name=model_name, model_type="local", device=device, **kwargs)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens

    def load(self) -> None:
        """Load model and processor with broad compatibility fallbacks."""
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoModelForVision2Seq,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError:
            raise ImportError("Please install: pip install transformers torch")

        logger.info(f"Loading {self.model_name}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        # Try the most specific multimodal loaders first, then fall back.
        last_error = None
        for model_cls in (
            AutoModelForImageTextToText,
            AutoModelForVision2Seq,
            AutoModelForCausalLM,
        ):
            try:
                self.model = model_cls.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                )
                self._is_loaded = True
                logger.info(f"Loaded {self.model_name} with {model_cls.__name__}")
                return
            except Exception as e:
                last_error = e

        raise RuntimeError(f"Failed to load {self.model_name}: {last_error}")

    def _build_inputs(self, img: Image.Image, prompt: str):
        """Create processor inputs with a chat-template-first strategy."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = prompt
        if hasattr(self.processor, "apply_chat_template"):
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = prompt

        try:
            inputs = self.processor(
                text=[text],
                images=[img],
                return_tensors="pt",
                padding=True,
            )
        except Exception:
            # Fallback for processors expecting non-list image/text arguments.
            inputs = self.processor(
                text=text,
                images=img,
                return_tensors="pt",
            )

        return inputs.to(self.device)

    @staticmethod
    def _decode_generated(processor, inputs, outputs) -> str:
        """Decode only newly generated tokens when possible."""
        try:
            if hasattr(inputs, "input_ids"):
                generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            else:
                generated_ids = outputs
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception:
            return ""

    def generate_report(
        self,
        image: Union[Image.Image, str],
        prompt: Optional[str] = None,
        **kwargs,
    ) -> ModelOutput:
        """Generate a report or image-grounded answer."""
        if not self._is_loaded:
            self.load()

        img = self.preprocess_image(image)
        prompt = prompt or "Generate a detailed radiology report for this image."

        inputs = self._build_inputs(img, prompt)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        text = self._decode_generated(self.processor, inputs, outputs)

        return ModelOutput(text=text)

    def answer_question(
        self,
        image: Union[Image.Image, str],
        question: str,
        **kwargs,
    ) -> ModelOutput:
        """Answer an image-grounded question."""
        if not self._is_loaded:
            self.load()

        img = self.preprocess_image(image)
        vqa_prompt = self.format_vqa_prompt(question)

        inputs = self._build_inputs(img, vqa_prompt)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )
        text = self._decode_generated(self.processor, inputs, outputs)

        return ModelOutput(text=text)
