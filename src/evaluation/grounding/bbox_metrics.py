"""
Bounding Box Evaluation Metrics

IoU and other metrics for evaluating grounding/localization.
"""

from typing import Dict, List
from src.evaluation.base_evaluator import BaseEvaluator


class BBoxEvaluator(BaseEvaluator):
    """Evaluator for bounding box localization accuracy."""
    
    def __init__(self, iou_thresholds: List[float] = [0.25, 0.5, 0.75], **kwargs):
        super().__init__(name="bbox", **kwargs)
        self.iou_thresholds = iou_thresholds
    
    def compute(
        self,
        predictions: List[Dict],
        references: List[Dict],
        **kwargs
    ) -> Dict[str, float]:
        """Compute bounding box metrics."""
        ious = []
        
        for pred, ref in zip(predictions, references):
            if pred and ref:
                iou = self._compute_iou(pred, ref)
                ious.append(iou)
        
        if not ious:
            return {"mean_iou": 0}
        
        results = {"mean_iou": sum(ious) / len(ious)}
        
        # Compute precision at different IoU thresholds
        for thresh in self.iou_thresholds:
            hits = sum(1 for iou in ious if iou >= thresh)
            results[f"precision@{thresh}"] = hits / len(ious)
        
        return results
    
    def _compute_iou(self, pred: Dict, ref: Dict) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(pred["x_min"], ref["x_min"])
        y1 = max(pred["y_min"], ref["y_min"])
        x2 = min(pred["x_max"], ref["x_max"])
        y2 = min(pred["y_max"], ref["y_max"])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        pred_area = (pred["x_max"] - pred["x_min"]) * (pred["y_max"] - pred["y_min"])
        ref_area = (ref["x_max"] - ref["x_min"]) * (ref["y_max"] - ref["y_min"])
        
        union = pred_area + ref_area - intersection
        
        return intersection / union if union > 0 else 0
