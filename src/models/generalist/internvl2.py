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
        model_name: str = "OpenGVLab/InternVL2-2B",
        device: str = "cuda",
        load_in_4bit: bool = False,
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
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        is_2b = "2B" in self.model_name or "2b" in self.model_name
        
        if self.load_in_4bit and not is_2b:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
            self.model = AutoModel.from_pretrained(
                self.model_name,
                **load_kwargs,
            ).eval()
        else:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                **load_kwargs,
            ).eval().cuda()
        
        self._is_loaded = True
        logger.info(f"Loaded {self.model_name}")

    
    def _build_pixel_values(self, img: Image.Image):
        """Build pixel_values tensor for InternVL2."""
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        pixel_values = transform(img).unsqueeze(0).to(torch.bfloat16).cuda()
        return pixel_values
    
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
        
        pixel_values = self._build_pixel_values(img)
        generation_config = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
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
        vqa_prompt = self.format_vqa_prompt(question)
        prompt = f"<image>\n{vqa_prompt}"
        
        pixel_values = self._build_pixel_values(img)
        generation_config = {"max_new_tokens": 50, "do_sample": False}
        
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
        )
        
        return ModelOutput(text=response)
