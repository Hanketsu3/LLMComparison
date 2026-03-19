"""
HuggingFace IU X-Ray Dataset Loader

Downloads and wraps the IU X-Ray dataset from HuggingFace for Report Generation (RRG).
Real chest X-ray images with real radiology reports.

HuggingFace source: IAMJB/IU-X-Ray-radiology-report-dataset (or similar)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)


class HFIUXRayDataset:
    """
    IU X-Ray dataset loaded directly from HuggingFace.
    
    Contains real chest X-ray images paired with real radiology reports
    (findings + impression). No local disk setup required.
    
    Usage:
        dataset = HFIUXRayDataset(split="test", max_samples=50)
        for item in dataset:
            print(item["findings"], item["impression"])
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
        
        logger.info(f"Downloading IU X-Ray ({split}) from HuggingFace...")
        
        # Try multiple known HuggingFace sources for IU X-Ray
        hf_sources = [
            "IAMJB/IU-X-Ray-radiology-report-dataset",
            "iu_xray",
        ]
        
        self._hf_dataset = None
        for source in hf_sources:
            try:
                self._hf_dataset = load_dataset(source, split=split)
                logger.info(f"Loaded from {source}")
                break
            except Exception as e:
                logger.warning(f"Could not load from {source}: {e}")
                continue
        
        if self._hf_dataset is None:
            raise RuntimeError(
                f"Could not load IU X-Ray from any source. Tried: {hf_sources}. "
                "Please check your internet connection or use VQA-RAD dataset instead."
            )
        
        if max_samples is not None:
            self._hf_dataset = self._hf_dataset.select(range(min(max_samples, len(self._hf_dataset))))
        
        # Detect column names
        cols = self._hf_dataset.column_names
        self._image_col = next((c for c in cols if "image" in c.lower()), None)
        self._findings_col = next((c for c in cols if "finding" in c.lower()), None)
        self._impression_col = next((c for c in cols if "impression" in c.lower()), None)
        self._report_col = next((c for c in cols if "report" in c.lower()), None)
        
        logger.info(f"Loaded IU X-Ray {split}: {len(self._hf_dataset)} samples (columns: {cols})")
    
    def __len__(self) -> int:
        return len(self._hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a sample dict with image, findings, impression, report."""
        sample = self._hf_dataset[idx]
        
        # Get image
        img = None
        if self._image_col and sample.get(self._image_col) is not None:
            img_data = sample[self._image_col]
            if isinstance(img_data, Image.Image):
                img = img_data.convert("RGB")
            else:
                try:
                    img = Image.open(img_data).convert("RGB")
                except Exception:
                    img = None
            
            if img is not None and self.image_size:
                img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Get text fields
        findings = sample.get(self._findings_col, "") if self._findings_col else ""
        impression = sample.get(self._impression_col, "") if self._impression_col else ""
        report = sample.get(self._report_col, "") if self._report_col else ""
        
        # Build full report if not directly available
        if not report and (findings or impression):
            parts = []
            if findings:
                parts.append(f"FINDINGS: {findings}")
            if impression:
                parts.append(f"IMPRESSION: {impression}")
            report = "\n".join(parts)
        
        return {
            "image": img,
            "image_path": None,
            "findings": findings or "",
            "impression": impression or "",
            "report": report or "",
            "study_id": str(idx),
            "split": self.split,
        }
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    @property
    def task(self) -> str:
        return "rrg"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        return {
            "num_samples": len(self._hf_dataset),
            "split": self.split,
            "source": "HuggingFace IU X-Ray",
            "columns": self._hf_dataset.column_names,
        }
