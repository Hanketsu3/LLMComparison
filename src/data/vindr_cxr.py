"""
VinDr-CXR Dataset Loader

Vietnamese Chest X-Ray dataset with localization annotations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image

from src.data.base_dataset import BaseRadiologyDataset, RadiologyItem

logger = logging.getLogger(__name__)


class VinDrCXRDataset(BaseRadiologyDataset):
    """
    VinDr-CXR dataset for abnormality detection and localization.
    
    Contains 18,000 chest X-ray images with radiologist annotations
    for 22 local labels (lung lesions) and 6 global labels.
    """
    
    FINDING_LABELS = [
        "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
        "Clavicle fracture", "Consolidation", "Edema", "Emphysema",
        "Enlarged PA", "ILD", "Infiltration", "Lung Opacity", "Lung cavity",
        "Lung cyst", "Mediastinal shift", "Nodule/Mass", "Pleural effusion",
        "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis",
        "Rib fracture", "Other lesion"
    ]
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "test",
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        load_images: bool = True,
    ):
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            max_samples=max_samples,
            image_size=image_size,
            load_images=load_images,
        )
        
        logger.info(f"Loaded VinDr-CXR {split} split with {len(self.samples)} samples")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load VinDr-CXR annotations."""
        # Load annotation CSV
        if self.split == "train":
            annotation_file = self.data_dir / "annotations" / "image_labels_train.csv"
        else:
            annotation_file = self.data_dir / "annotations" / "image_labels_test.csv"
        
        if not annotation_file.exists():
            # Try alternative location
            annotation_file = self.data_dir / f"{self.split}_annotations.csv"
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        df = pd.read_csv(annotation_file)
        
        # Group by image
        samples = []
        for image_id, group in df.groupby("image_id"):
            findings = []
            bboxes = []
            
            for _, row in group.iterrows():
                label = row.get("class_name", row.get("label", ""))
                if label and label != "No finding":
                    findings.append(label)
                    
                    # Extract bounding box if available
                    if all(col in row for col in ["x_min", "y_min", "x_max", "y_max"]):
                        bbox = {
                            "x_min": row["x_min"],
                            "y_min": row["y_min"],
                            "x_max": row["x_max"],
                            "y_max": row["y_max"],
                        }
                        bboxes.append(bbox)
            
            sample = {
                "image_id": image_id,
                "image_path": str(self.data_dir / self.split / f"{image_id}.png"),
                "findings": findings,
                "bboxes": bboxes,
            }
            samples.append(sample)
        
        return samples
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load a chest X-ray image."""
        path = Path(image_path)
        
        # Try different extensions
        if not path.exists():
            for ext in [".png", ".jpg", ".jpeg", ".dicom"]:
                alt_path = path.with_suffix(ext)
                if alt_path.exists():
                    path = alt_path
                    break
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(path).convert("RGB")
        
        if self.image_size:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _create_item(self, annotation: Dict[str, Any]) -> RadiologyItem:
        """Create a RadiologyItem from annotation."""
        return RadiologyItem(
            study_id=annotation["image_id"],
            image_id=annotation["image_id"],
            image_path=annotation["image_path"],
            findings_list=annotation["findings"],
            bounding_boxes=annotation["bboxes"],
            modality="xray",
            split=self.split,
        )
    
    @property
    def task(self) -> str:
        return "grounding"
