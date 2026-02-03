"""
VQA-RAD Dataset Loader

Visual Question Answering in Radiology dataset.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from src.data.base_dataset import BaseRadiologyDataset, RadiologyItem

logger = logging.getLogger(__name__)


class VQARADDataset(BaseRadiologyDataset):
    """
    VQA-RAD dataset for visual question answering in radiology.
    
    Contains 315 images and 3,515 question-answer pairs covering
    head, chest, and abdomen imaging.
    """
    
    QUESTION_TYPES = {
        "closed": ["yes/no", "binary", "closed"],
        "open": ["what", "where", "how", "when", "which", "open"],
    }
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "test",
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        load_images: bool = True,
        question_types: Optional[List[str]] = None,  # "closed", "open", or both
    ):
        self.question_type_filter = question_types
        
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            max_samples=max_samples,
            image_size=image_size,
            load_images=load_images,
        )
        
        logger.info(f"Loaded VQA-RAD {split} split with {len(self.samples)} samples")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load VQA-RAD annotations."""
        # Try different annotation file names
        possible_files = [
            "VQA_RAD_Dataset_Public.json",
            "vqa_rad.json",
            f"{self.split}.json",
        ]
        
        annotation_file = None
        for fname in possible_files:
            path = self.data_dir / fname
            if path.exists():
                annotation_file = path
                break
        
        if annotation_file is None:
            raise FileNotFoundError(f"No annotation file found in {self.data_dir}")
        
        with open(annotation_file, "r") as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict) and "data" in data:
            samples = data["data"]
        elif isinstance(data, list):
            samples = data
        else:
            samples = list(data.values()) if isinstance(data, dict) else []
        
        # Filter by split if split info is available
        if samples and "split" in samples[0]:
            samples = [s for s in samples if s.get("split") == self.split]
        
        # Filter by question type
        if self.question_type_filter:
            filtered = []
            for sample in samples:
                q_type = sample.get("answer_type", sample.get("question_type", ""))
                for filter_type in self.question_type_filter:
                    if filter_type.lower() in q_type.lower():
                        filtered.append(sample)
                        break
            samples = filtered
        
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
    
    def _classify_question_type(self, question: str, answer: str) -> str:
        """Classify question as closed or open-ended."""
        answer_lower = answer.lower().strip()
        
        # Check if answer is yes/no
        if answer_lower in ["yes", "no"]:
            return "closed"
        
        # Check question patterns
        question_lower = question.lower()
        for open_pattern in ["what", "where", "how many", "which", "describe"]:
            if question_lower.startswith(open_pattern):
                return "open"
        
        # Default based on answer length
        if len(answer.split()) <= 2:
            return "closed"
        
        return "open"
    
    def _create_item(self, annotation: Dict[str, Any]) -> RadiologyItem:
        """Create a RadiologyItem from annotation."""
        # Extract fields (handle different naming conventions)
        image_name = annotation.get("image_name", annotation.get("image", ""))
        image_path = str(self.data_dir / "images" / image_name)
        
        question = annotation.get("question", "")
        answer = annotation.get("answer", "")
        
        # Determine question type
        q_type = annotation.get("answer_type", annotation.get("question_type", ""))
        if not q_type:
            q_type = self._classify_question_type(question, answer)
        elif "close" in q_type.lower():
            q_type = "closed"
        elif "open" in q_type.lower():
            q_type = "open"
        
        return RadiologyItem(
            study_id=annotation.get("qid", str(hash(question))),
            image_id=image_name,
            image_path=image_path,
            question=question,
            answer=answer,
            question_type=q_type,
            modality=annotation.get("modality", "unknown"),
            split=self.split,
            metadata={
                "image_organ": annotation.get("image_organ", ""),
                "phrase_type": annotation.get("phrase_type", ""),
            }
        )
    
    @property
    def task(self) -> str:
        return "vqa"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        stats = super().get_statistics()
        
        # Count question types
        type_counts = {"closed": 0, "open": 0, "unknown": 0}
        
        for sample in self.samples:
            q_type = sample.get("answer_type", sample.get("question_type", "unknown"))
            if "close" in str(q_type).lower():
                type_counts["closed"] += 1
            elif "open" in str(q_type).lower():
                type_counts["open"] += 1
            else:
                type_counts["unknown"] += 1
        
        stats["question_type_distribution"] = type_counts
        
        return stats
