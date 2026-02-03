"""
SLAKE Dataset Loader

A Semantically-Labeled Knowledge-Enhanced Dataset for Medical VQA.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from src.data.base_dataset import BaseRadiologyDataset, RadiologyItem

logger = logging.getLogger(__name__)


class SLAKEDataset(BaseRadiologyDataset):
    """
    SLAKE (Semantically-Labeled Knowledge-Enhanced) dataset.
    
    A large-scale bilingual (English & Chinese) medical VQA dataset
    with semantic labels and medical knowledge annotations.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "test",
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        load_images: bool = True,
        language: str = "en",  # "en" or "zh"
        question_types: Optional[List[str]] = None,
    ):
        self.language = language
        self.question_type_filter = question_types
        
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            max_samples=max_samples,
            image_size=image_size,
            load_images=load_images,
        )
        
        logger.info(f"Loaded SLAKE {split} split ({language}) with {len(self.samples)} samples")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load SLAKE annotations."""
        # Determine annotation file based on split
        split_files = {
            "train": "train.json",
            "val": "validate.json",
            "test": "test.json",
        }
        
        annotation_file = self.data_dir / split_files.get(self.split, f"{self.split}.json")
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, "r") as f:
            data = json.load(f)
        
        # Filter by language
        samples = []
        for item in data:
            if self.language == "en":
                question = item.get("question", "")
                answer = item.get("answer", "")
            else:
                question = item.get("question_zh", item.get("question", ""))
                answer = item.get("answer_zh", item.get("answer", ""))
            
            item["question"] = question
            item["answer"] = answer
            
            # Apply question type filter
            if self.question_type_filter:
                q_type = item.get("answer_type", "")
                if not any(ft.lower() in q_type.lower() for ft in self.question_type_filter):
                    continue
            
            samples.append(item)
        
        return samples
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load a medical image."""
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        image = Image.open(path).convert("RGB")
        
        if self.image_size:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _create_item(self, annotation: Dict[str, Any]) -> RadiologyItem:
        """Create a RadiologyItem from annotation."""
        image_name = annotation.get("img_name", annotation.get("image", ""))
        image_path = str(self.data_dir / "imgs" / image_name)
        
        # Determine question type
        q_type = annotation.get("answer_type", "")
        if "close" in q_type.lower():
            q_type = "closed"
        elif "open" in q_type.lower():
            q_type = "open"
        
        return RadiologyItem(
            study_id=str(annotation.get("qid", hash(annotation["question"]))),
            image_id=image_name,
            image_path=image_path,
            question=annotation["question"],
            answer=str(annotation["answer"]),
            question_type=q_type,
            modality=annotation.get("modality", ""),
            split=self.split,
            metadata={
                "content_type": annotation.get("content_type", ""),
                "q_lang": self.language,
                "organ": annotation.get("location", ""),
            }
        )
    
    @property
    def task(self) -> str:
        return "vqa"
