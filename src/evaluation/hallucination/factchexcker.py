"""FactCheXcker evaluator for finding-level hallucination analysis."""

import logging
import re
from typing import Dict, List, Set, Tuple

from src.evaluation.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class FactCheXckerEvaluator(BaseEvaluator):
    """Evaluator for detecting hallucinated findings in radiology reports."""

    _NEGATION_CUES = ("no", "without", "absent", "negative for", "free of")

    _FINDING_PATTERNS = {
        "cardiomegaly": [r"cardiomegal", r"enlarged\s+heart"],
        "pleural_effusion": [r"pleural\s+effusion", r"effusion"],
        "pneumonia": [r"pneumonia", r"bronchopneumonia"],
        "pneumothorax": [r"pneumothorax", r"ptx"],
        "atelectasis": [r"atelecta(?:sis|tic)"],
        "consolidation": [r"consolidat"],
        "edema": [r"edema", r"oedema", r"vascular\s+congestion"],
        "mass": [r"mass"],
        "nodule": [r"nodule"],
        "fracture": [r"fracture"],
        "emphysema": [r"emphysema"],
        "fibrosis": [r"fibrosis", r"fibrotic"],
        "infiltrate": [r"infiltrate"],
    }
    
    def __init__(self, **kwargs):
        super().__init__(name="factchexcker", **kwargs)
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute hallucination and factuality metrics at finding-level."""
        total_hallucinations = 0
        total_contradictions = 0
        total_pred_findings = 0
        total_correct = 0
        hallucination_scores: List[float] = []

        for pred, ref in zip(predictions, references):
            pred_pos, pred_neg = self._extract_findings_with_polarity(pred)
            ref_pos, ref_neg = self._extract_findings_with_polarity(ref)

            hallucinations = pred_pos - ref_pos
            contradictions = (pred_pos & ref_neg) | (pred_neg & ref_pos)
            correct_findings = pred_pos & ref_pos

            denominator = len(pred_pos) + len(contradictions)
            sample_hall_score = (len(hallucinations) + len(contradictions)) / denominator if denominator > 0 else 0.0
            hallucination_scores.append(sample_hall_score)

            total_hallucinations += len(hallucinations)
            total_contradictions += len(contradictions)
            total_pred_findings += len(pred_pos)
            total_correct += len(correct_findings)

        avg_hallucination_score = (
            sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0
        )
        factual_precision = (total_correct / total_pred_findings) if total_pred_findings > 0 else 0.0

        # Recall uses reference-positive findings observed over all samples.
        total_ref_positive = sum(len(self._extract_findings_with_polarity(ref)[0]) for ref in references)
        factual_recall = (total_correct / total_ref_positive) if total_ref_positive > 0 else 0.0
        denom = factual_precision + factual_recall
        factual_f1 = (2 * factual_precision * factual_recall / denom) if denom > 0 else 0.0

        return {
            "hallucination_rate": avg_hallucination_score,
            "hallucination_score": avg_hallucination_score,
            "factual_precision": factual_precision,
            "factual_recall": factual_recall,
            "factual_f1": factual_f1,
            "total_hallucinations": total_hallucinations,
            "total_contradictions": total_contradictions,
            "total_findings": total_pred_findings,
        }

    def _extract_findings_with_polarity(self, text: str) -> Tuple[Set[str], Set[str]]:
        """Extract positive and negated findings from text."""
        text_lower = text.lower()
        positive: Set[str] = set()
        negative: Set[str] = set()

        for finding, patterns in self._FINDING_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    if self._is_negated(text_lower, match.start()):
                        negative.add(finding)
                    else:
                        positive.add(finding)

        return positive, negative

    def _is_negated(self, text_lower: str, start_idx: int) -> bool:
        window = text_lower[max(0, start_idx - 40):start_idx]
        return any(cue in window for cue in self._NEGATION_CUES)
