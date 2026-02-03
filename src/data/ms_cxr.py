"""
MS-CXR Dataset Loader

Microsoft Chest X-Ray dataset with phrase grounding annotations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from src.data.base_dataset import BaseRadiologyDataset, RadiologyItem

logger = logging.getLogger(__name__)


class MSCXRDataset(BaseRadiologyDataset):
    """
    MS-CXR (Microsoft Chest X-Ray) dataset for phrase grounding.
    
    Contains bounding box annotations for phrases in radiology reports,
    linking text descriptions to image regions.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "test",
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        load_images: bool = True,
        mimic_cxr_path: Optional[str] = None,  # Path to MIMIC-CXR images
    ):
        self.mimic_cxr_path = Path(mimic_cxr_path) if mimic_cxr_path else None
        
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            max_samples=max_samples,
            image_size=image_size,
            load_images=load_images,
        )
        
        logger.info(f"Loaded MS-CXR {split} split with {len(self.samples)} samples")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load MS-CXR annotations with bounding boxes."""
        # Look for annotation file
        annotation_file = self.data_dir / f"MS_CXR_Local_Alignment_{self.split.capitalize()}.json"
        
        if not annotation_file.exists():
            # Try alternative names
            for fname in ["annotations.json", f"{self.split}.json", "ms_cxr.json"]:
                alt_path = self.data_dir / fname
                if alt_path.exists():
                    annotation_file = alt_path
                    break
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found in {self.data_dir}")
        
        with open(annotation_file, "r") as f:
            data = json.load(f)
        
        return data if isinstance(data, list) else list(data.values())
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load a chest X-ray image."""
        path = Path(image_path)
        
        if not path.exists() and self.mimic_cxr_path:
            # Try to find in MIMIC-CXR
            # Expected format: pXX/pXXXXXXXX/sXXXXXXXX/XXXXX.jpg
            path = self.mimic_cxr_path / "files" / path.name
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(path).convert("RGB")
        
        if self.image_size:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _normalize_bbox(
        self, 
        bbox: Dict[str, float], 
        original_size: Tuple[int, int]
    ) -> Dict[str, float]:
        """Normalize bounding box coordinates to [0, 1] range."""
        w, h = original_size
        return {
            "x_min": bbox["x_min"] / w,
            "y_min": bbox["y_min"] / h,
            "x_max": bbox["x_max"] / w,
            "y_max": bbox["y_max"] / h,
        }
    
    def _create_item(self, annotation: Dict[str, Any]) -> RadiologyItem:
        """Create a RadiologyItem from annotation."""
        # Extract image info
        image_path = annotation.get("image_path", annotation.get("dicom_id", ""))
        if not image_path:
            image_path = f"{annotation.get('subject_id', '')}/{annotation.get('study_id', '')}.jpg"
        
        # Extract grounding data
        findings_list = []
        bounding_boxes = []
        
        # Handle different annotation formats
        if "label_text" in annotation:
            # Single finding format
            findings_list.append(annotation["label_text"])
            bbox = {
                "x_min": annotation.get("x", 0),
                "y_min": annotation.get("y", 0),
                "x_max": annotation.get("x", 0) + annotation.get("w", 0),
                "y_max": annotation.get("y", 0) + annotation.get("h", 0),
            }
            bounding_boxes.append(bbox)
        elif "annotations" in annotation:
            # Multiple findings format
            for ann in annotation["annotations"]:
                findings_list.append(ann.get("phrase", ann.get("label", "")))
                bbox = ann.get("bbox", {})
                if isinstance(bbox, list):
                    bbox = {
                        "x_min": bbox[0],
                        "y_min": bbox[1],
                        "x_max": bbox[2],
                        "y_max": bbox[3],
                    }
                bounding_boxes.append(bbox)
        
        return RadiologyItem(
            study_id=str(annotation.get("study_id", hash(image_path))),
            image_id=annotation.get("dicom_id", Path(image_path).stem),
            image_path=str(self.data_dir / image_path) if not Path(image_path).is_absolute() else image_path,
            findings_list=findings_list,
            bounding_boxes=bounding_boxes,
            modality="xray",
            split=self.split,
            metadata={
                "category": annotation.get("category", ""),
            }
        )
    
    @property
    def task(self) -> str:
        return "grounding"
