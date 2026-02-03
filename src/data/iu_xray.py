"""
IU X-Ray Dataset Loader

Indiana University Chest X-Ray Collection.
"""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from src.data.base_dataset import BaseRadiologyDataset, RadiologyItem

logger = logging.getLogger(__name__)


class IUXRayDataset(BaseRadiologyDataset):
    """
    IU X-Ray (Indiana University Chest X-Ray) dataset.
    
    Contains 7,470 chest X-ray images paired with radiology reports.
    https://openi.nlm.nih.gov/
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
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            max_samples=max_samples,
            image_size=image_size,
            load_images=load_images,
        )
        logger.info(f"Loaded IU X-Ray {split} split with {len(self.samples)} samples")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load IU X-Ray annotations."""
        # Look for annotation file
        annotation_file = self.data_dir / "annotation.json"
        
        if annotation_file.exists():
            return self._load_from_json(annotation_file)
        
        # Try to load from XML reports
        reports_dir = self.data_dir / "reports"
        if reports_dir.exists():
            return self._load_from_xml(reports_dir)
        
        raise FileNotFoundError(
            f"No annotation file found in {self.data_dir}. "
            "Expected 'annotation.json' or 'reports/' directory."
        )
    
    def _load_from_json(self, annotation_file: Path) -> List[Dict[str, Any]]:
        """Load annotations from JSON file."""
        with open(annotation_file, "r") as f:
            data = json.load(f)
        
        # Filter by split
        samples = [s for s in data if s.get("split", "train") == self.split]
        return samples
    
    def _load_from_xml(self, reports_dir: Path) -> List[Dict[str, Any]]:
        """Load annotations from XML report files."""
        samples = []
        
        for xml_file in reports_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Extract study info
                study_id = xml_file.stem
                
                # Extract images
                images = []
                for img in root.findall(".//parentImage"):
                    img_id = img.get("id")
                    if img_id:
                        images.append(img_id)
                
                # Extract report text
                findings = ""
                impression = ""
                
                for abstract in root.findall(".//AbstractText"):
                    label = abstract.get("Label", "").lower()
                    text = abstract.text or ""
                    
                    if "finding" in label:
                        findings = text
                    elif "impression" in label:
                        impression = text
                
                for img_id in images:
                    sample = {
                        "study_id": study_id,
                        "image_id": img_id,
                        "image_path": str(self.data_dir / "images" / f"{img_id}.png"),
                        "findings": findings,
                        "impression": impression,
                    }
                    samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"Failed to parse {xml_file}: {e}")
        
        return samples
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load an X-ray image."""
        path = Path(image_path)
        
        if not path.exists():
            # Try different extensions
            for ext in [".png", ".jpg", ".jpeg"]:
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
        findings = annotation.get("findings", "")
        impression = annotation.get("impression", "")
        full_report = f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"
        
        return RadiologyItem(
            study_id=annotation["study_id"],
            image_id=annotation["image_id"],
            image_path=annotation["image_path"],
            findings=findings,
            impression=impression,
            full_report=full_report,
            modality="xray",
            split=self.split,
        )
    
    @property
    def task(self) -> str:
        return "rrg"
