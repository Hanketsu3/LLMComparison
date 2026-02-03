"""
PadChest Dataset - OOD generalization testing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from PIL import Image
from src.data.base_dataset import BaseRadiologyDataset, RadiologyItem

logger = logging.getLogger(__name__)


class PadChestDataset(BaseRadiologyDataset):
    """PadChest dataset for out-of-distribution testing."""
    
    def __init__(self, data_dir: Union[str, Path], split: str = "test", **kwargs):
        super().__init__(data_dir=data_dir, split=split, **kwargs)
        logger.info(f"Loaded PadChest {split} with {len(self.samples)} samples")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        ann_file = self.data_dir / "PADCHEST_chest_x_ray_images_labels_160K.csv"
        if not ann_file.exists():
            ann_file = self.data_dir / "padchest_labels.csv"
        df = pd.read_csv(ann_file, low_memory=False)
        
        samples = []
        for _, row in df.iterrows():
            samples.append({
                "image_id": row.get("ImageID", ""),
                "image_path": str(self.data_dir / "images" / row.get("ImageID", "")),
                "report": row.get("Report", ""),
            })
        return samples
    
    def _load_image(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        if self.image_size:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        return image
    
    def _create_item(self, annotation: Dict[str, Any]) -> RadiologyItem:
        return RadiologyItem(
            study_id=annotation["image_id"],
            image_id=annotation["image_id"], 
            image_path=annotation["image_path"],
            full_report=annotation.get("report", ""),
            modality="xray",
            split=self.split,
        )
    
    @property
    def task(self) -> str:
        return "rrg"
