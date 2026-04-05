"""CheXbert F1 evaluator for chest X-ray report labeling."""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.evaluation.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class CheXbertF1Evaluator(BaseEvaluator):
    """CheXbert-based F1 score for chest X-ray report evaluation.

    Uses a real CheXbert model when available, otherwise falls back to a
    clinically informed ruleset with synonym and negation handling.
    """
    
    LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
        "Pneumonia", "Pneumothorax", "Support Devices"
    ]
    
    _NEGATION_CUES = ("no", "without", "absent", "negative for", "free of")

    _LABEL_PATTERNS = {
        "Atelectasis": [r"atelecta(?:sis|tic)"],
        "Cardiomegaly": [r"cardiomegal", r"enlarged\s+heart", r"cardiac\s+silhouette\s+is\s+enlarged"],
        "Consolidation": [r"consolidat"],
        "Edema": [r"edema", r"oedema", r"pulmonary\s+vascular\s+congestion"],
        "Enlarged Cardiomediastinum": [r"cardiomediastinal\s+silhouette\s+is\s+enlarged", r"mediastinal\s+widening"],
        "Fracture": [r"fracture", r"broken\s+rib"],
        "Lung Lesion": [r"lung\s+lesion", r"pulmonary\s+lesion"],
        "Lung Opacity": [r"opacity", r"opacities", r"airspace\s+opacity", r"infiltrate"],
        "No Finding": [r"no\s+acute\s+cardiopulmonary\s+abnormalit", r"no\s+acute\s+disease", r"no\s+finding"],
        "Pleural Effusion": [r"pleural\s+effusion", r"effusion"],
        "Pleural Other": [r"pleural\s+thickening", r"pleural\s+plaque", r"pleural\s+disease"],
        "Pneumonia": [r"pneumonia", r"bronchopneumonia"],
        "Pneumothorax": [r"pneumothorax", r"ptx"],
        "Support Devices": [r"tube", r"line", r"catheter", r"pacemaker", r"stent", r"support\s+device"],
    }

    def __init__(self, **kwargs):
        super().__init__(name="chexbert_f1", **kwargs)
        self.model: Optional[Any] = None
        self._model_is_available = False
    
    def _load_model(self) -> None:
        """Try loading a CheXbert model backend.

        The external package APIs vary by version, so this loader remains
        defensive and falls back cleanly when unavailable.
        """
        try:
            from chexbert import CheXbert  # type: ignore

            self.model = CheXbert()
            self._model_is_available = True
            logger.info("Loaded CheXbert model from chexbert package")
        except Exception as exc:
            self.model = None
            self._model_is_available = False
            logger.warning("CheXbert model unavailable (%s). Using rule-based fallback.", exc)
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute CheXbert micro/macro F1 and default ``chexbert_f1``."""
        if self.model is None and not self._model_is_available:
            self._load_model()

        if self._model_is_available and self.model is not None:
            pred_labels = self._predict_with_model(predictions)
            ref_labels = self._predict_with_model(references)
        else:
            pred_labels = [self._extract_labels_fallback(text) for text in predictions]
            ref_labels = [self._extract_labels_fallback(text) for text in references]

        micro_f1, macro_f1 = self._compute_micro_macro_f1(pred_labels, ref_labels)
        return {
            "chexbert_micro_f1": float(micro_f1),
            "chexbert_macro_f1": float(macro_f1),
            "chexbert_f1": float(macro_f1),
            "chexbert_fallback_used": 0.0 if self._model_is_available else 1.0,
        }

    def _predict_with_model(self, reports: List[str]) -> List[Set[str]]:
        """Run CheXbert model if available and normalize output labels."""
        if self.model is None:
            return [set() for _ in reports]

        try:
            raw = self.model(reports)
        except Exception as exc:
            logger.warning("CheXbert inference failed (%s). Falling back to rules.", exc)
            return [self._extract_labels_fallback(text) for text in reports]

        normalized: List[Set[str]] = []
        for item in raw:
            labels = self._normalize_model_output_item(item)
            normalized.append(labels)
        return normalized

    def _normalize_model_output_item(self, item: Any) -> Set[str]:
        """Normalize possible CheXbert output formats into positive labels."""
        labels: Set[str] = set()

        if isinstance(item, dict):
            for label in self.LABELS:
                value = item.get(label)
                if self._is_positive_label_value(value):
                    labels.add(label)
            return labels

        if isinstance(item, list):
            for entry in item:
                if isinstance(entry, str) and entry in self.LABELS:
                    labels.add(entry)
                elif isinstance(entry, dict):
                    lbl = entry.get("label")
                    if lbl in self.LABELS and self._is_positive_label_value(entry.get("value", 1)):
                        labels.add(lbl)
            return labels

        return labels

    @staticmethod
    def _is_positive_label_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return float(value) > 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "positive", "present", "yes"}
        return bool(value)

    def _extract_labels_fallback(self, text: str) -> Set[str]:
        """Rule-based extraction with simple local negation handling."""
        text_lower = text.lower()
        found: Set[str] = set()

        for label, patterns in self._LABEL_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    if not self._is_negated(text_lower, match.start()):
                        found.add(label)
                        break
                if label in found:
                    break

        return found

    def _is_negated(self, text_lower: str, start_idx: int) -> bool:
        """Detect local negation cues in a short left context window."""
        window = text_lower[max(0, start_idx - 40):start_idx]
        return any(cue in window for cue in self._NEGATION_CUES)

    def _compute_micro_macro_f1(
        self,
        pred_labels: List[Set[str]],
        ref_labels: List[Set[str]],
    ) -> Tuple[float, float]:
        """Compute multilabel micro/macro F1 without external dependencies."""
        if not pred_labels or not ref_labels:
            return 0.0, 0.0

        total_tp = 0
        total_fp = 0
        total_fn = 0
        per_label_f1: List[float] = []

        for label in self.LABELS:
            tp = fp = fn = 0
            for pred, ref in zip(pred_labels, ref_labels):
                pred_has = label in pred
                ref_has = label in ref
                if pred_has and ref_has:
                    tp += 1
                elif pred_has and not ref_has:
                    fp += 1
                elif (not pred_has) and ref_has:
                    fn += 1

            total_tp += tp
            total_fp += fp
            total_fn += fn
            denom = (2 * tp + fp + fn)
            per_label_f1.append((2 * tp / denom) if denom > 0 else 0.0)

        micro_denom = (2 * total_tp + total_fp + total_fn)
        micro_f1 = (2 * total_tp / micro_denom) if micro_denom > 0 else 0.0
        macro_f1 = sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0
        return micro_f1, macro_f1
