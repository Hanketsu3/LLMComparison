"""
FactCheXcker - Hallucination Detection for Radiology Reports

Detects fabricated findings not present in the source image.
"""

import logging
from typing import Dict, List
from src.evaluation.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class FactCheXckerEvaluator(BaseEvaluator):
    """Evaluator for detecting hallucinated findings in radiology reports."""
    
    COMMON_FINDINGS = [
        "cardiomegaly", "pleural effusion", "pneumonia", "pneumothorax",
        "atelectasis", "consolidation", "edema", "mass", "nodule",
        "fracture", "emphysema", "fibrosis", "infiltrate"
    ]
    
    def __init__(self, **kwargs):
        super().__init__(name="factchexcker", **kwargs)
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute hallucination rate."""
        total_hallucinations = 0
        total_findings = 0
        hallucination_rates = []
        
        for pred, ref in zip(predictions, references):
            pred_findings = self._extract_findings(pred)
            ref_findings = self._extract_findings(ref)
            
            # Findings in prediction but not in reference = hallucinations
            hallucinations = pred_findings - ref_findings
            
            if len(pred_findings) > 0:
                rate = len(hallucinations) / len(pred_findings)
                hallucination_rates.append(rate)
            
            total_hallucinations += len(hallucinations)
            total_findings += len(pred_findings)
        
        avg_rate = sum(hallucination_rates) / len(hallucination_rates) if hallucination_rates else 0
        
        return {
            "hallucination_rate": avg_rate,
            "total_hallucinations": total_hallucinations,
            "total_findings": total_findings,
        }
    
    def _extract_findings(self, text: str) -> set:
        """Extract clinical findings from text."""
        text_lower = text.lower()
        found = set()
        
        for finding in self.COMMON_FINDINGS:
            if finding in text_lower:
                found.add(finding)
        
        return found
