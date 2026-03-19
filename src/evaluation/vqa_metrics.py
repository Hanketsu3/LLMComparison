"""
VQA Evaluation Metrics

Accuracy and other metrics for Visual Question Answering.
Separates closed-ended (yes/no) and open-ended accuracy per VQA-RAD leaderboard.
"""

import re
from typing import Dict, List, Union
from src.evaluation.base_evaluator import BaseEvaluator


class VQAAccuracyEvaluator(BaseEvaluator):
    """
    VQA accuracy evaluator with closed/open separation.
    
    Metrics reported:
    - accuracy: overall accuracy
    - closed_accuracy: yes/no question accuracy
    - open_accuracy: open-ended question accuracy
    """
    
    # Common articles and filler words to strip for matching
    STRIP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "it", "there"}
    
    def __init__(self, **kwargs):
        super().__init__(name="vqa_accuracy", **kwargs)
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute VQA accuracy (overall, closed, open)."""
        correct = 0
        total = 0
        closed_correct = 0
        closed_total = 0
        open_correct = 0
        open_total = 0
        
        for pred, ref in zip(predictions, references):
            is_closed = self._is_closed_question(ref)
            match = self._is_correct(pred, ref)
            
            if match:
                correct += 1
            total += 1
            
            if is_closed:
                closed_total += 1
                if match:
                    closed_correct += 1
            else:
                open_total += 1
                if match:
                    open_correct += 1
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "closed_accuracy": closed_correct / closed_total if closed_total > 0 else 0.0,
            "open_accuracy": open_correct / open_total if open_total > 0 else 0.0,
            "closed_total": closed_total,
            "open_total": open_total,
            "total": total,
        }
    
    def _is_closed_question(self, reference: str) -> bool:
        """Check if this is a yes/no (closed-ended) question."""
        ref = str(reference).lower().strip().rstrip(".")
        return ref in ["yes", "no"]
    
    def _normalize(self, text: str) -> str:
        """Normalize answer text for comparison."""
        if not text:
            return ""
        
        text = str(text).lower().strip()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove articles and filler words
        words = text.split()
        words = [w for w in words if w not in self.STRIP_WORDS]
        
        return " ".join(words).strip()
    
    def _is_correct(self, pred: str, ref: str) -> bool:
        """Check if answer is correct with smart matching."""
        if not pred or not ref:
            return False
        
        pred_raw = str(pred).lower().strip().rstrip(".")
        ref_raw = str(ref).lower().strip().rstrip(".")
        
        # --- Closed-ended (yes/no) matching ---
        if ref_raw in ["yes", "no"]:
            # Extract yes/no from potentially verbose model output
            pred_clean = pred_raw.replace(",", " ").replace(".", " ")
            first_word = pred_clean.split()[0] if pred_clean.split() else ""
            
            # Direct match
            if first_word == ref_raw:
                return True
            
            # Check if model says the answer anywhere
            if ref_raw == "yes" and any(w in pred_clean.split() for w in ["yes", "correct", "affirmative"]):
                if not any(w in pred_clean.split() for w in ["no", "not", "negative"]):
                    return True
            if ref_raw == "no" and any(w in pred_clean.split() for w in ["no", "not", "negative", "incorrect"]):
                if not any(w in pred_clean.split() for w in ["yes"]):
                    return True
            
            return False
        
        # --- Open-ended matching ---
        pred_norm = self._normalize(pred_raw)
        ref_norm = self._normalize(ref_raw)
        
        # Exact match after normalization
        if pred_norm == ref_norm:
            return True
        
        # Reference contained in prediction (model might say "the answer is X")
        if ref_norm and ref_norm in pred_norm:
            return True
        
        # Prediction contained in reference
        if pred_norm and pred_norm in ref_norm:
            return True
        
        # Token recall: what fraction of reference tokens appear in prediction?
        ref_tokens = set(ref_norm.split())
        pred_tokens = set(pred_norm.split())
        if ref_tokens and len(ref_tokens & pred_tokens) / len(ref_tokens) >= 0.8:
            return True
        
        return False
