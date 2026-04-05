"""BERTScore evaluator for semantic similarity in report generation."""

import logging
from typing import Dict, List

from src.evaluation.base_evaluator import BaseEvaluator


logger = logging.getLogger(__name__)


class BERTScoreEvaluator(BaseEvaluator):
    """BERTScore metric wrapper.

    Returns precision/recall/f1 means across samples.
    """

    def __init__(self, model_type: str = "distilbert-base-uncased", lang: str = "en", **kwargs):
        super().__init__(name="bertscore", **kwargs)
        self.model_type = model_type
        self.lang = lang

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        non_empty = [(p, r) for p, r in zip(predictions, references) if p and p.strip()]
        if not non_empty:
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "bertscore_fallback_used": 0.0,
            }
        predictions, references = zip(*non_empty)
        predictions, references = list(predictions), list(references)

        try:
            from evaluate import load

            bertscore = load("bertscore")
            result = bertscore.compute(
                predictions=predictions,
                references=references,
                model_type=self.model_type,
                lang=self.lang,
            )
            p_vals = result.get("precision", [])
            r_vals = result.get("recall", [])
            f_vals = result.get("f1", [])
            return {
                "bertscore_precision": float(sum(p_vals) / len(p_vals)) if p_vals else 0.0,
                "bertscore_recall": float(sum(r_vals) / len(r_vals)) if r_vals else 0.0,
                "bertscore_f1": float(sum(f_vals) / len(f_vals)) if f_vals else 0.0,
                "bertscore_fallback_used": 0.0,
            }
        except Exception as exc:
            logger.warning("BERTScore computation failed: %s", exc)
            return {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "bertscore_fallback_used": 1.0,
            }
