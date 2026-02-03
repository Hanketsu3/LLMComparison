"""
VQA Evaluation Metrics

Accuracy and other metrics for Visual Question Answering.
"""

from typing import Dict, List, Union
from src.evaluation.base_evaluator import BaseEvaluator


class VQAAccuracyEvaluator(BaseEvaluator):
    """Exact match accuracy for VQA."""
    
    def __init__(self, **kwargs):
        super().__init__(name="vqa_accuracy", **kwargs)
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute VQA accuracy."""
        correct = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            if self._is_correct(pred, ref):
                correct += 1
            total += 1
        
        if total == 0:
            return {"accuracy": 0.0}
            
        return {"accuracy": correct / total}
    
    def _is_correct(self, pred: str, ref: str) -> bool:
        """Check if answer is correct (exact match after normalization)."""
        if not pred or not ref:
            return False
            
        # Basic normalization
        pred = str(pred).lower().strip().rstrip(".")
        ref = str(ref).lower().strip().rstrip(".")
        
        # Handle yes/no variants
        if ref in ["yes", "no"]:
            return pred.startswith(ref)
            
        return pred == ref
