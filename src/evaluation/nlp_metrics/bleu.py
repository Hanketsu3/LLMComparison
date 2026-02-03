"""BLEU Score Evaluator."""

from typing import Dict, List
from src.evaluation.base_evaluator import BaseEvaluator


class BLEUEvaluator(BaseEvaluator):
    """BLEU score for text generation evaluation."""
    
    def __init__(self, max_order: int = 4, **kwargs):
        super().__init__(name="bleu", **kwargs)
        self.max_order = max_order
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute BLEU score."""
        try:
            from evaluate import load
            bleu = load("bleu")
        except ImportError:
            from nltk.translate.bleu_score import corpus_bleu
            # Tokenize
            pred_tokens = [p.split() for p in predictions]
            ref_tokens = [[r.split()] for r in references]
            score = corpus_bleu(ref_tokens, pred_tokens)
            return {"bleu": score}
        
        # Format references for HuggingFace evaluate
        refs = [[r] for r in references]
        result = bleu.compute(predictions=predictions, references=refs)
        
        return {
            "bleu": result.get("bleu", 0),
            "bleu_1": result.get("precisions", [0])[0] if result.get("precisions") else 0,
            "bleu_4": result.get("precisions", [0, 0, 0, 0])[3] if len(result.get("precisions", [])) > 3 else 0,
        }
