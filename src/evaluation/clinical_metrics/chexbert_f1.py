"""
CheXbert F1 Score Evaluator

Measures accuracy of chest X-ray finding classifications.
"""

import logging
from typing import Dict, List
from src.evaluation.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class CheXbertF1Evaluator(BaseEvaluator):
    """CheXbert-based F1 score for chest X-ray report evaluation."""
    
    LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
        "Pneumonia", "Pneumothorax", "Support Devices"
    ]
    
    def __init__(self, **kwargs):
        super().__init__(name="chexbert_f1", **kwargs)
        self.model = None
    
    def _load_model(self):
        """Load CheXbert model."""
        logger.info("Loading CheXbert model...")
        # CheXbert model loading would go here
        # For now, using placeholder
        self.model = True
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute CheXbert F1 score."""
        if self.model is None:
            self._load_model()
        
        # Extract labels from predictions and references
        pred_labels = [self._extract_labels(p) for p in predictions]
        ref_labels = [self._extract_labels(r) for r in references]
        
        # Compute per-label and micro/macro F1
        from sklearn.metrics import f1_score
        import numpy as np
        
        # Convert to binary arrays
        pred_array = np.array([[1 if l in p else 0 for l in self.LABELS] for p in pred_labels])
        ref_array = np.array([[1 if l in r else 0 for l in self.LABELS] for r in ref_labels])
        
        micro_f1 = f1_score(ref_array.flatten(), pred_array.flatten(), zero_division=0)
        macro_f1 = f1_score(ref_array, pred_array, average='macro', zero_division=0)
        
        return {
            "chexbert_micro_f1": float(micro_f1),
            "chexbert_macro_f1": float(macro_f1),
        }
    
    def _extract_labels(self, text: str) -> List[str]:
        """Extract finding labels from report text."""
        found = []
        text_lower = text.lower()
        
        for label in self.LABELS:
            if label.lower() in text_lower:
                found.append(label)
        
        return found
