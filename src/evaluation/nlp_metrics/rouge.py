"""ROUGE Score Evaluator."""

from typing import Dict, List
from src.evaluation.base_evaluator import BaseEvaluator


class ROUGEEvaluator(BaseEvaluator):
    """ROUGE score for text generation evaluation."""
    
    def __init__(self, **kwargs):
        super().__init__(name="rouge", **kwargs)
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute ROUGE scores."""
        try:
            from evaluate import load
            rouge = load("rouge")
        except ImportError:
            return {"rouge_l": 0}
        
        result = rouge.compute(predictions=predictions, references=references)
        
        return {
            "rouge_1": result.get("rouge1", 0),
            "rouge_2": result.get("rouge2", 0),
            "rouge_l": result.get("rougeL", 0),
        }
