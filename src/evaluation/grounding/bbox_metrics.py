"""
Bounding Box Evaluation Metrics

IoU and other metrics for evaluating grounding/localization.
"""

from typing import Dict, List, Tuple
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

    def compute_set_metrics(self, predictions: List[Dict], references: List[Dict]) -> Dict[str, float]:
        """Compute one-to-one greedy matched metrics for multi-bbox grounding."""
        if not predictions or not references:
            return {
                "mean_iou": 0.0,
                "precision@0.25": 0.0,
                "precision@0.5": 0.0,
                "precision@0.75": 0.0,
                "recall@0.5": 0.0,
            }

        matches = self._greedy_match(predictions, references)
        ious = [iou for _, _, iou in matches]
        mean_iou = sum(ious) / len(ious) if ious else 0.0

        precisions = {}
        for thresh in self.iou_thresholds:
            hits = sum(1 for iou in ious if iou >= thresh)
            precisions[f"precision@{thresh}"] = hits / len(predictions) if predictions else 0.0

        hits_05 = sum(1 for iou in ious if iou >= 0.5)
        recall_05 = hits_05 / len(references) if references else 0.0

        return {
            "mean_iou": mean_iou,
            **precisions,
            "recall@0.5": recall_05,
        }

    def _greedy_match(self, predictions: List[Dict], references: List[Dict]) -> List[Tuple[int, int, float]]:
        """Greedily match prediction/reference boxes by highest IoU without reuse."""
        candidates: List[Tuple[float, int, int]] = []
        for pred_idx, pred_box in enumerate(predictions):
            for ref_idx, ref_box in enumerate(references):
                iou = self._compute_iou(pred_box, ref_box)
                candidates.append((iou, pred_idx, ref_idx))

        candidates.sort(reverse=True, key=lambda x: x[0])

        used_pred = set()
        used_ref = set()
        matches: List[Tuple[int, int, float]] = []
        for iou, pred_idx, ref_idx in candidates:
            if pred_idx in used_pred or ref_idx in used_ref:
                continue
            used_pred.add(pred_idx)
            used_ref.add(ref_idx)
            matches.append((pred_idx, ref_idx, iou))

        return matches
