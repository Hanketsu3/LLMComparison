"""
MIMIC-CXR Dataset Loader

Large-scale chest X-ray dataset with free-text radiology reports.
https://physionet.org/content/mimic-cxr/2.0.0/
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image

from src.data.base_dataset import BaseRadiologyDataset, RadiologyItem

logger = logging.getLogger(__name__)


class MIMICCXRDataset(BaseRadiologyDataset):
    """
    MIMIC-CXR dataset for radiology report generation.
    
    The dataset contains 377,110 chest X-ray images and 227,827 imaging studies
    from 65,379 patients at Beth Israel Deaconess Medical Center.
    
    Directory structure expected:
        mimic-cxr/
        ├── files/
        │   ├── p10/
        │   │   ├── p10000032/
        │   │   │   ├── s50414267/
        │   │   │   │   ├── *.jpg
        │   │   │   │   └── ...
        ├── mimic-cxr-2.0.0-metadata.csv
        ├── mimic-cxr-2.0.0-split.csv
        └── mimic-cxr-reports/
            └── files/
    """
    
    VIEWS = {
        "frontal": ["PA", "AP"],
        "lateral": ["LATERAL", "LL"],
    }
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "test",
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        load_images: bool = True,
        views: Optional[List[str]] = None,  # "frontal", "lateral", or specific views
        use_jpg: bool = True,  # Use JPG instead of DICOM
    ):
        """
        Initialize MIMIC-CXR dataset.
        
        Args:
            data_dir: Path to MIMIC-CXR root directory
            split: Dataset split ("train", "validate", "test")
            transform: Optional image transform
            max_samples: Maximum samples to load
            image_size: Target image size
            load_images: Whether to load images
            views: Filter by view type
            use_jpg: Use JPG images (faster) instead of DICOM
        """
        self.views = views or ["frontal"]
        self.use_jpg = use_jpg
        
        super().__init__(
            data_dir=data_dir,
            split=split,
            transform=transform,
            max_samples=max_samples,
            image_size=image_size,
            load_images=load_images,
        )
        
        logger.info(f"Loaded MIMIC-CXR {split} split with {len(self.samples)} samples")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load MIMIC-CXR annotations and metadata."""
        # Load split file
        split_file = self.data_dir / "mimic-cxr-2.0.0-split.csv"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        splits_df = pd.read_csv(split_file)
        
        # Load metadata
        metadata_file = self.data_dir / "mimic-cxr-2.0.0-metadata.csv"
        if metadata_file.exists():
            metadata_df = pd.read_csv(metadata_file)
        else:
            metadata_df = None
            logger.warning("Metadata file not found, proceeding without view filtering")
        
        # Filter by split
        split_mapping = {"train": "train", "val": "validate", "test": "test"}
        target_split = split_mapping.get(self.split, self.split)
        split_studies = splits_df[splits_df["split"] == target_split]
        
        # Join with metadata if available
        if metadata_df is not None:
            merged = split_studies.merge(
                metadata_df[["dicom_id", "subject_id", "study_id", "ViewPosition"]],
                on=["dicom_id", "subject_id", "study_id"],
                how="inner"
            )
            
            # Filter by view
            view_filter = []
            for view_type in self.views:
                if view_type in self.VIEWS:
                    view_filter.extend(self.VIEWS[view_type])
                else:
                    view_filter.append(view_type)
            
            if view_filter:
                merged = merged[merged["ViewPosition"].isin(view_filter)]
        else:
            merged = split_studies
        
        # Build samples list
        samples = []
        for _, row in merged.iterrows():
            subject_id = str(row["subject_id"])
            study_id = str(row["study_id"])
            dicom_id = row["dicom_id"]
            
            # Build image path
            subject_prefix = f"p{subject_id[:2]}"
            image_path = (
                self.data_dir / "files" / subject_prefix / 
                f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"
            )
            
            # Build report path
            report_path = (
                self.data_dir / "mimic-cxr-reports" / "files" / 
                subject_prefix / f"p{subject_id}" / f"s{study_id}.txt"
            )
            
            sample = {
                "subject_id": subject_id,
                "study_id": study_id,
                "dicom_id": dicom_id,
                "image_path": str(image_path),
                "report_path": str(report_path),
                "view": row.get("ViewPosition", "unknown"),
            }
            samples.append(sample)
        
        return samples
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load a chest X-ray image."""
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        image = Image.open(path).convert("RGB")
        
        # Resize if needed
        if self.image_size:
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _load_report(self, report_path: str) -> Tuple[str, str, str]:
        """
        Load and parse a radiology report.
        
        Returns:
            Tuple of (findings, impression, full_report)
        """
        path = Path(report_path)
        
        if not path.exists():
            return "", "", ""
        
        with open(path, "r") as f:
            text = f.read()
        
        # Parse sections
        findings = self._extract_section(text, "FINDINGS")
        impression = self._extract_section(text, "IMPRESSION")
        
        return findings, impression, text
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a section from a radiology report."""
        import re
        
        # Pattern to match section headers
        pattern = rf"{section_name}[:\s]*\n?(.*?)(?=\n[A-Z]+[:\s]|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _create_item(self, annotation: Dict[str, Any]) -> RadiologyItem:
        """Create a RadiologyItem from annotation."""
        findings, impression, full_report = self._load_report(annotation["report_path"])
        
        return RadiologyItem(
            study_id=annotation["study_id"],
            image_id=annotation["dicom_id"],
            image_path=annotation["image_path"],
            findings=findings,
            impression=impression,
            full_report=full_report,
            view=annotation["view"],
            modality="xray",
            split=self.split,
            metadata={
                "subject_id": annotation["subject_id"],
                "report_path": annotation["report_path"],
            }
        )
    
    @property
    def task(self) -> str:
        return "rrg"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        stats = super().get_statistics()
        
        # Count views
        view_counts = {}
        for sample in self.samples:
            view = sample.get("view", "unknown")
            view_counts[view] = view_counts.get(view, 0) + 1
        
        stats["view_distribution"] = view_counts
        
        return stats
