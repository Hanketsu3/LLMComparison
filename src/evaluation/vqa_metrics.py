"""VQA evaluation metrics for open-ended and closed-answer tasks."""

import re
from typing import Dict, List

from src.evaluation.base_evaluator import BaseEvaluator


STRIP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "it", "there"}


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    if not text:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    words = [word for word in text.split() if word not in STRIP_WORDS]
    return " ".join(words).strip()


def token_f1_score(prediction: str, reference: str) -> Dict[str, float]:
    """Compute token-level precision, recall, and F1 on normalized answers."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}

    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in set(pred_counts).intersection(ref_counts):
        overlap += min(pred_counts[token], ref_counts[token])

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def _edit_distance(left: str, right: str) -> int:
    """Compute Levenshtein distance for short answer strings."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_row = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current_row = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (left_char != right_char)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def anls_score(prediction: str, reference: str, threshold: float = 0.5) -> float:
    """Compute ANLS for a single prediction/reference pair."""
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)

    if not pred_norm and not ref_norm:
        return 1.0
    if not pred_norm or not ref_norm:
        return 0.0

    max_len = max(len(pred_norm), len(ref_norm))
    if max_len == 0:
        return 1.0

    similarity = 1.0 - (_edit_distance(pred_norm, ref_norm) / max_len)
    if similarity < threshold:
        return 0.0
    return max(0.0, similarity)


class VQASoftMatchEvaluator(BaseEvaluator):
    """Soft VQA answer matching with token-F1 and ANLS."""

    def __init__(self, **kwargs):
        super().__init__(name="vqa_soft_match", **kwargs)

    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs,
    ) -> Dict[str, float]:
        token_precision_values: List[float] = []
        token_recall_values: List[float] = []
        token_f1_values: List[float] = []
        anls_values: List[float] = []

        for prediction, reference in zip(predictions, references):
            token_scores = token_f1_score(prediction, reference)
            token_precision_values.append(token_scores["token_precision"])
            token_recall_values.append(token_scores["token_recall"])
            token_f1_values.append(token_scores["token_f1"])
            anls_values.append(anls_score(prediction, reference))

        total = len(token_f1_values)
        return {
            "token_precision": sum(token_precision_values) / total if total > 0 else 0.0,
            "token_recall": sum(token_recall_values) / total if total > 0 else 0.0,
            "token_f1": sum(token_f1_values) / total if total > 0 else 0.0,
            "anls": sum(anls_values) / total if total > 0 else 0.0,
            "total": total,
        }


VQAAccuracyEvaluator = VQASoftMatchEvaluator
