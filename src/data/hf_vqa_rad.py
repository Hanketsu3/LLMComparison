"""
HuggingFace VQA-RAD Dataset Loader

Downloads and wraps the VQA-RAD dataset from HuggingFace for direct use
without any local disk setup. Real patient data, real questions, real answers.

HuggingFace source: flaviagiammarino/vqa-rad
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)


class HFVQARADDataset:
    """
    VQA-RAD dataset loaded directly from HuggingFace.
    
    No local disk setup required — downloads automatically.
    Contains real radiology images with expert-annotated Q&A pairs.
    
    Usage:
        dataset = HFVQARADDataset(split="test", max_samples=100)
        for item in dataset:
            print(item["question"], item["answer"])
            img = item["image"]  # PIL Image
    """
    
    def __init__(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.split = split
        self.max_samples = max_samples
        self.image_size = image_size
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install: pip install datasets")
        
        logger.info(f"Downloading VQA-RAD ({split}) from HuggingFace...")
        self._hf_dataset = load_dataset("flaviagiammarino/vqa-rad", split=split)
        
        if max_samples is not None:
            self._hf_dataset = self._hf_dataset.select(range(min(max_samples, len(self._hf_dataset))))
        
        logger.info(f"Loaded VQA-RAD {split}: {len(self._hf_dataset)} samples")
    
    def __len__(self) -> int:
        return len(self._hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a sample dict with image, question, answer, etc."""
        sample = self._hf_dataset[idx]
        
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        
        if self.image_size:
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        
        return {
            "image": img,
            "image_path": None,  # no file on disk
            "question": sample["question"],
            "answer": sample["answer"],
            "study_id": str(idx),
            "split": self.split,
        }
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def save_images_to_disk(self, output_dir: str) -> List[str]:
        """
        Save all dataset images to disk (for models requiring file paths).
        Returns list of saved file paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        paths = []
        for i in range(len(self)):
            sample = self._hf_dataset[i]
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            
            path = out / f"vqa_rad_{self.split}_{i:04d}.png"
            img.save(str(path))
            paths.append(str(path))
        
        logger.info(f"Saved {len(paths)} images to {output_dir}")
        return paths
    
    @property
    def task(self) -> str:
        return "vqa"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        closed = sum(1 for s in self._hf_dataset if str(s.get("answer", "")).lower().strip() in ["yes", "no"])
        return {
            "num_samples": len(self._hf_dataset),
            "split": self.split,
            "source": "HuggingFace: flaviagiammarino/vqa-rad",
            "closed_questions": closed,
            "open_questions": len(self._hf_dataset) - closed,
        }
