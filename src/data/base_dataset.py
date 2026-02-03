"""
Base Dataset class for radiology datasets.

Provides a unified interface for all radiology datasets used in the comparison study.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class RadiologyItem:
    """Data class representing a single radiology sample."""
    
    # Identifiers
    study_id: str
    image_id: str
    
    # Image data
    image: Optional[Image.Image] = None
    image_path: Optional[str] = None
    
    # Report data (for RRG task)
    findings: Optional[str] = None
    impression: Optional[str] = None
    full_report: Optional[str] = None
    
    # VQA data
    question: Optional[str] = None
    answer: Optional[str] = None
    question_type: Optional[str] = None  # "closed" or "open"
    
    # Grounding data
    findings_list: Optional[List[str]] = None
    bounding_boxes: Optional[List[Dict[str, float]]] = None  # [{"x_min", "y_min", "x_max", "y_max"}]
    
    # Metadata
    view: Optional[str] = None  # "frontal", "lateral", "PA", "AP"
    modality: Optional[str] = None  # "xray", "ct", "mri"
    split: Optional[str] = None  # "train", "val", "test"
    metadata: Optional[Dict[str, Any]] = None


class BaseRadiologyDataset(Dataset, ABC):
    """
    Abstract base class for radiology datasets.
    
    All dataset implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "test",
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        load_images: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory of the dataset
            split: Dataset split ("train", "val", "test")
            transform: Optional image transform function
            max_samples: Maximum number of samples to load (for debugging)
            image_size: Target image size for resizing
            load_images: Whether to load images into memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.image_size = image_size
        self.load_images = load_images
        
        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Load dataset annotations
        self.samples = self._load_annotations()
        
        # Apply max_samples limit
        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]
    
    @abstractmethod
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load dataset annotations from disk.
        
        Returns:
            List of annotation dictionaries
        """
        pass
    
    @abstractmethod
    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        pass
    
    @abstractmethod
    def _create_item(self, annotation: Dict[str, Any]) -> RadiologyItem:
        """
        Create a RadiologyItem from an annotation dictionary.
        
        Args:
            annotation: Annotation dictionary
            
        Returns:
            RadiologyItem instance
        """
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> RadiologyItem:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            RadiologyItem instance
        """
        annotation = self.samples[idx]
        item = self._create_item(annotation)
        
        # Load image if requested
        if self.load_images and item.image_path:
            item.image = self._load_image(item.image_path)
            
            # Apply transform if provided
            if self.transform is not None:
                item.image = self.transform(item.image)
        
        return item
    
    def get_task_subset(self, task: str) -> "BaseRadiologyDataset":
        """
        Get a subset of the dataset for a specific task.
        
        Args:
            task: Task name ("rrg", "vqa", "grounding")
            
        Returns:
            Filtered dataset
        """
        # Subclasses should override this if needed
        return self
    
    @property
    def task(self) -> str:
        """Return the primary task this dataset supports."""
        return "rrg"  # Default to report generation
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "num_samples": len(self.samples),
            "split": self.split,
            "data_dir": str(self.data_dir),
        }
        return stats
    
    def collate_fn(self, batch: List[RadiologyItem]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of RadiologyItem instances
            
        Returns:
            Collated batch dictionary
        """
        collated = {
            "study_ids": [item.study_id for item in batch],
            "image_ids": [item.image_id for item in batch],
            "images": [item.image for item in batch if item.image is not None],
            "image_paths": [item.image_path for item in batch],
        }
        
        # Add task-specific fields
        if batch[0].full_report is not None:
            collated["reports"] = [item.full_report for item in batch]
            collated["findings"] = [item.findings for item in batch]
            collated["impressions"] = [item.impression for item in batch]
        
        if batch[0].question is not None:
            collated["questions"] = [item.question for item in batch]
            collated["answers"] = [item.answer for item in batch]
            collated["question_types"] = [item.question_type for item in batch]
        
        if batch[0].bounding_boxes is not None:
            collated["bounding_boxes"] = [item.bounding_boxes for item in batch]
            collated["findings_list"] = [item.findings_list for item in batch]
        
        return collated
