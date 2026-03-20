"""
Unified Sample Schema for Radiology Benchmark

All datasets must standardize their output to this RadiologySample dataclass
to ensure compatibility across different experiments and evaluation metrics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image


@dataclass
class BoundingBox:
    """Standardized bounding box representation."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: Optional[str] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            **({"label": self.label} if self.label else {}),
            **({"confidence": self.confidence} if self.confidence is not None else {}),
        }


@dataclass
class RadiologySample:
    """
    Unified schema for radiology samples across all datasets and tasks.
    
    This dataclass ensures that different data loaders produce compatible
    outputs, reducing hidden field name bugs and making evaluation/metrics
    agnostic to dataset origin.
    """
    
    # ========================================================================
    # IDENTIFIERS (Required)
    # ========================================================================
    sample_id: str
    """Unique identifier within dataset (e.g., 'MIMIC_12345_frontal')"""
    
    dataset_name: str
    """Dataset origin (e.g., 'MIMIC_CXR', 'VQA_RAD', 'MS_CXR')"""
    
    split: str
    """Data split: 'train', 'val', 'test'"""
    
    # ========================================================================
    # IMAGE DATA
    # ========================================================================
    image_path: Optional[str] = None
    """Path to single image file"""
    
    image_paths: Optional[List[str]] = None
    """Paths to multiple images (for multi-view tasks)"""
    
    image: Optional[Image.Image] = None
    """Loaded PIL Image (optional, for in-memory storage)"""
    
    # ========================================================================
    # TASK 1: REPORT GENERATION (RRG)
    # ========================================================================
    report_reference: Optional[str] = None
    """Full ground-truth radiology report"""
    
    findings_reference: Optional[str] = None
    """Structured finding section (optional split)"""
    
    impression_reference: Optional[str] = None
    """Structured impression section (optional split)"""
    
    # ========================================================================
    # TASK 2: VISUAL QUESTION ANSWERING (VQA)
    # ========================================================================
    question: Optional[str] = None
    """VQA question"""
    
    answer_reference: Optional[str] = None
    """Ground-truth answer"""
    
    question_type: Optional[str] = None
    """Question category: 'closed', 'open', 'yes_no', 'counting', etc."""
    
    # ========================================================================
    # TASK 3: GROUNDING / LOCALIZATION
    # ========================================================================
    findings_list: Optional[List[str]] = None
    """Findings to localize (e.g., ['pneumonia', 'cardiomediastinal mass'])"""
    
    bounding_boxes: Optional[List[BoundingBox]] = None
    """Ground-truth bounding boxes for findings"""
    
    # ========================================================================
    # METADATA
    # ========================================================================
    view: Optional[str] = None
    """Image view: 'frontal', 'lateral', 'PA', 'AP', 'lateral_down', etc."""
    
    views: Optional[List[str]] = None
    """Multiple views if multi-view"""
    
    age: Optional[int] = None
    """Patient age in years"""
    
    gender: Optional[str] = None
    """Patient gender: 'M', 'F', or None if unknown"""
    
    pathology_labels: Optional[Dict[str, float]] = None
    """Semantic labels: {'pneumonia': 0.8, 'cardiomegaly': 0.3, ...}
    Values are binary (0/1) or confidence scores (0-1)"""
    
    modality: Optional[str] = None
    """Image modality: 'xray', 'ct', 'mri', 'ultrasound', etc."""
    
    body_part: Optional[str] = None
    """Body part: 'chest', 'abdomen', 'head', 'extremity', etc."""
    
    # ========================================================================
    # ADDITIONAL CONTEXT
    # ========================================================================
    study_date: Optional[str] = None
    """Date of study in YYYY-MM-DD format"""
    
    report_date: Optional[str] = None
    """Date of report generation"""
    
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    """Arbitrary additional metadata dict"""
    
    # ========================================================================
    # VALIDATION & UTILITIES
    # ========================================================================
    
    def get_image(self) -> Optional[Image.Image]:
        """Get image, loading from disk if necessary."""
        if self.image is not None:
            return self.image
        if self.image_path and Path(self.image_path).exists():
            return Image.open(self.image_path)
        return None
    
    def is_valid_for_task(self, task: str) -> bool:
        """Check if sample has required fields for a task."""
        task = task.lower()
        if task == "rrg" or task == "report_generation":
            return (self.image_path or self.image_path or self.image) and self.report_reference
        elif task == "vqa" or task == "visual_question_answering":
            return (self.image_path or self.image_paths or self.image) and self.question and self.answer_reference
        elif task == "grounding" or task == "localization":
            return (self.image_path or self.image_paths or self.image) and self.bounding_boxes
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def has_image(self) -> bool:
        """Check if sample has at least one image."""
        return bool(self.image_path or self.image_paths or self.image)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding PIL Image for JSON serialization)."""
        result = {}
        result["sample_id"] = self.sample_id
        result["dataset_name"] = self.dataset_name
        result["split"] = self.split
        
        if self.image_path:
            result["image_path"] = self.image_path
        if self.image_paths:
            result["image_paths"] = self.image_paths
        
        if self.report_reference:
            result["report_reference"] = self.report_reference
        if self.findings_reference:
            result["findings_reference"] = self.findings_reference
        if self.impression_reference:
            result["impression_reference"] = self.impression_reference
        
        if self.question:
            result["question"] = self.question
        if self.answer_reference:
            result["answer_reference"] = self.answer_reference
        if self.question_type:
            result["question_type"] = self.question_type
        
        if self.findings_list:
            result["findings_list"] = self.findings_list
        if self.bounding_boxes:
            result["bounding_boxes"] = [bbox.to_dict() for bbox in self.bounding_boxes]
        
        if self.view:
            result["view"] = self.view
        if self.views:
            result["views"] = self.views
        if self.age is not None:
            result["age"] = self.age
        if self.gender:
            result["gender"] = self.gender
        if self.pathology_labels:
            result["pathology_labels"] = self.pathology_labels
        if self.modality:
            result["modality"] = self.modality
        if self.body_part:
            result["body_part"] = self.body_part
        
        if self.study_date:
            result["study_date"] = self.study_date
        if self.report_date:
            result["report_date"] = self.report_date
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RadiologySample":
        """Create sample from dictionary."""
        # Parse bounding boxes
        bboxes = None
        if "bounding_boxes" in data and data["bounding_boxes"]:
            bboxes = [
                BoundingBox(
                    x_min=bbox["x_min"],
                    y_min=bbox["y_min"],
                    x_max=bbox["x_max"],
                    y_max=bbox["y_max"],
                    label=bbox.get("label"),
                    confidence=bbox.get("confidence"),
                )
                for bbox in data["bounding_boxes"]
            ]
        
        return cls(
            sample_id=data.get("sample_id"),
            dataset_name=data.get("dataset_name"),
            split=data.get("split"),
            image_path=data.get("image_path"),
            image_paths=data.get("image_paths"),
            report_reference=data.get("report_reference"),
            findings_reference=data.get("findings_reference"),
            impression_reference=data.get("impression_reference"),
            question=data.get("question"),
            answer_reference=data.get("answer_reference"),
            question_type=data.get("question_type"),
            findings_list=data.get("findings_list"),
            bounding_boxes=bboxes,
            view=data.get("view"),
            views=data.get("views"),
            age=data.get("age"),
            gender=data.get("gender"),
            pathology_labels=data.get("pathology_labels"),
            modality=data.get("modality"),
            body_part=data.get("body_part"),
            study_date=data.get("study_date"),
            report_date=data.get("report_date"),
            metadata=data.get("metadata", {}),
        )
