"""METEOR evaluator for report generation."""

from typing import Dict, List

from src.evaluation.base_evaluator import BaseEvaluator


class METEOREvaluator(BaseEvaluator):
    """METEOR metric using HuggingFace evaluate when available."""

    def __init__(self, **kwargs):
        super().__init__(name="meteor", **kwargs)

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        non_empty = [(p, r) for p, r in zip(predictions, references) if p and p.strip()]
        if not non_empty:
            return {"meteor": 0.0}
        predictions, references = zip(*non_empty)
        predictions, references = list(predictions), list(references)

        try:
            from evaluate import load

            meteor = load("meteor")
            result = meteor.compute(predictions=predictions, references=references)
            return {"meteor": float(result.get("meteor", 0.0))}
        except Exception:
            return {"meteor": 0.0}
