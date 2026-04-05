#!/usr/bin/env python3
"""
Unified experiment runner for radiology benchmark lanes.

Pipeline:
config -> dataset -> model -> inference -> sample metrics -> aggregate -> paired stats
"""

import argparse
import itertools
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.environment import EnvironmentManager, RuntimePreset
from src.data.schema import BoundingBox, RadiologySample
from src.evaluation.clinical_metrics.chexbert_f1 import CheXbertF1Evaluator
from src.evaluation.clinical_metrics.radgraph_f1 import RadGraphF1Evaluator
from src.evaluation.grounding.bbox_metrics import BBoxEvaluator
from src.evaluation.hallucination.factchexcker import FactCheXckerEvaluator
from src.evaluation.nlp_metrics.bertscore import BERTScoreEvaluator
from src.evaluation.nlp_metrics.bleu import BLEUEvaluator
from src.evaluation.nlp_metrics.meteor import METEOREvaluator
from src.evaluation.nlp_metrics.rouge import ROUGEEvaluator
from src.evaluation.vqa_metrics import VQAAccuracyEvaluator, anls_score, normalize_answer, token_f1_score
from src.utils.logging import setup_logging
from src.utils.model_registry import (
    DOMAIN_ADAPTIVE_MODELS,
    GENERALIST_MODELS,
    MAIN_MODELS,
    SPECIALIST_MODELS,
    ModelInfo,
    check_model_access,
    get_model_info,
    load_model,
)
from src.utils.result_writer import PredictionRecord, ResultWriter, SampleMetric
from src.utils.statistical_tests import StatisticalTester

logger = logging.getLogger(__name__)


SUPPORTED_DATASETS = ("hf_iu_xray", "hf_vqa_rad", "ms_cxr")
_EVALUATOR_CACHE: Dict[str, Any] = {}


def _get_evaluator(name: str):
    """Lazily initialize evaluators once per run."""
    if name not in _EVALUATOR_CACHE:
        if name == "bleu":
            _EVALUATOR_CACHE[name] = BLEUEvaluator()
        elif name == "rouge":
            _EVALUATOR_CACHE[name] = ROUGEEvaluator()
        elif name == "meteor":
            _EVALUATOR_CACHE[name] = METEOREvaluator()
        elif name == "bertscore":
            _EVALUATOR_CACHE[name] = BERTScoreEvaluator()
        elif name == "chexbert":
            _EVALUATOR_CACHE[name] = CheXbertF1Evaluator()
        elif name == "radgraph":
            _EVALUATOR_CACHE[name] = RadGraphF1Evaluator(reward_level="full")
        elif name == "hallucination":
            _EVALUATOR_CACHE[name] = FactCheXckerEvaluator()
        elif name == "vqa":
            _EVALUATOR_CACHE[name] = VQAAccuracyEvaluator()
        elif name == "bbox":
            _EVALUATOR_CACHE[name] = BBoxEvaluator()
        else:
            raise ValueError(f"Unknown evaluator: {name}")
    return _EVALUATOR_CACHE[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified radiology benchmark runner")
    parser.add_argument("--preset", default="gpu_24g", choices=[p.value for p in RuntimePreset])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["main"],
        help=(
            "Model selector: main, generalist, domain_adaptive, specialist,"
            " or explicit model names"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hf_iu_xray", "hf_vqa_rad"],
        help="Datasets to run (hf_iu_xray, hf_vqa_rad, ms_cxr)",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--ms-cxr-path", type=str, default=None)
    parser.add_argument("--skip-inaccessible", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-fallback-rate",
        type=float,
        default=1.0,
        help="Fail run if any *_fallback_used mean exceeds this ratio (0.0-1.0).",
    )
    parser.add_argument(
        "--strict-sample-validation",
        action="store_true",
        help="Fail immediately when a normalized sample does not satisfy task requirements.",
    )
    return parser.parse_args()


def _normalize_question_type(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    value = str(raw).strip().lower().replace("-", "_")
    if value in {"yes/no", "yes_no", "boolean", "closed", "close"}:
        return "closed"
    if value in {"open", "open_ended", "openended"}:
        return "open"
    if value in {"count", "counting", "numeric", "number"}:
        return "counting"
    return value


def validate_sample_for_task(sample: RadiologySample, task: str) -> Tuple[bool, Optional[str]]:
    """Validate normalized sample against task-required fields and bbox sanity."""
    try:
        if not sample.is_valid_for_task(task):
            return False, f"missing_required_fields_for_{task}"
    except Exception as exc:
        return False, f"validation_error:{exc}"

    if task == "grounding" and sample.bounding_boxes:
        for idx, box in enumerate(sample.bounding_boxes):
            if box.x_max <= box.x_min or box.y_max <= box.y_min:
                return False, f"invalid_bbox_geometry_at_index_{idx}"

    return True, None


def compute_fallback_rates(sample_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean fallback flags from sample-level metrics."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in sample_metrics:
        for key, value in row.items():
            if not key.endswith("_fallback_used"):
                continue
            if not isinstance(value, (int, float)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

    return {
        key: (sums[key] / counts[key]) if counts[key] > 0 else 0.0
        for key in sorted(sums.keys())
    }


def set_global_seed(seed: int) -> None:
    """Configure reproducible RNG state across common libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        logger.debug("NumPy seed setup skipped.")

    try:
        import importlib.util

        if importlib.util.find_spec("torch") is not None:
            torch_mod = __import__("torch")
            torch_mod.manual_seed(seed)
            if torch_mod.cuda.is_available():
                torch_mod.cuda.manual_seed_all(seed)
            try:
                torch_mod.backends.cudnn.deterministic = True
                torch_mod.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        logger.debug("Torch seed setup skipped.")


def get_git_commit_hash() -> Optional[str]:
    """Return HEAD commit hash when git metadata is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            return commit if commit else None
    except Exception:
        return None
    return None


def resolve_models(selectors: List[str]) -> List[str]:
    resolved: List[str] = []
    for selector in selectors:
        selector = selector.lower()
        if selector == "main":
            resolved.extend(MAIN_MODELS)
        elif selector == "generalist":
            resolved.extend(GENERALIST_MODELS)
        elif selector == "domain_adaptive":
            resolved.extend(DOMAIN_ADAPTIVE_MODELS)
        elif selector == "specialist":
            resolved.extend(SPECIALIST_MODELS)
        else:
            resolved.append(selector)

    unique = []
    seen = set()
    for model_name in resolved:
        if model_name not in seen:
            unique.append(model_name)
            seen.add(model_name)
    return unique


def _extract_images(sample: RadiologySample) -> Optional[Union[str, Any]]:
    if sample.image is not None:
        return sample.image
    if sample.image_path:
        return sample.image_path
    if sample.image_paths:
        return sample.image_paths[0]
    return None


def normalize_sample(raw: Any, dataset_name: str, split: str) -> RadiologySample:
    if isinstance(raw, RadiologySample):
        return raw

    if isinstance(raw, dict):
        bboxes = []
        raw_boxes = raw.get("bounding_boxes") or []
        for box in raw_boxes:
            if isinstance(box, BoundingBox):
                bboxes.append(box)
            elif isinstance(box, dict):
                bboxes.append(
                    BoundingBox(
                        x_min=float(box.get("x_min", 0.0)),
                        y_min=float(box.get("y_min", 0.0)),
                        x_max=float(box.get("x_max", 0.0)),
                        y_max=float(box.get("y_max", 0.0)),
                        label=box.get("label"),
                    )
                )

        return RadiologySample(
            sample_id=str(raw.get("sample_id") or raw.get("study_id") or raw.get("qid") or "unknown"),
            dataset_name=dataset_name,
            split=str(raw.get("split") or split),
            image_path=raw.get("image_path"),
            image=raw.get("image"),
            report_reference=raw.get("report") or raw.get("full_report") or raw.get("report_reference"),
            findings_reference=raw.get("findings") or raw.get("findings_reference"),
            impression_reference=raw.get("impression") or raw.get("impression_reference"),
            question=raw.get("question"),
            answer_reference=raw.get("answer") or raw.get("answer_reference"),
            question_type=_normalize_question_type(raw.get("question_type")),
            findings_list=raw.get("findings_list"),
            bounding_boxes=bboxes or None,
            view=raw.get("view"),
            modality=raw.get("modality") or "xray",
            body_part=raw.get("body_part") or "chest",
            metadata={"raw_keys": sorted(raw.keys())},
        )

    as_dict = vars(raw)
    return normalize_sample(as_dict, dataset_name=dataset_name, split=split)


def load_dataset(dataset_name: str, num_samples: Optional[int], ms_cxr_path: Optional[str]) -> Tuple[str, Iterable[Any]]:
    ds = dataset_name.lower()
    if ds not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {SUPPORTED_DATASETS}")

    if ds == "hf_iu_xray":
        from src.data import HFIUXRayDataset

        return "rrg", HFIUXRayDataset(split="test", max_samples=num_samples)
    if ds == "hf_vqa_rad":
        from src.data import HFVQARADDataset

        return "vqa", HFVQARADDataset(split="test", max_samples=num_samples)

    if not ms_cxr_path:
        raise ValueError("Dataset ms_cxr requires --ms-cxr-path")
    from src.data import MSCXRDataset

    return "grounding", MSCXRDataset(data_dir=ms_cxr_path, split="test", max_samples=num_samples)


def _compute_text_metrics(predicted_text: str, reference_text: str) -> Dict[str, Optional[float]]:
    pred_norm = normalize_answer(predicted_text)
    ref_norm = normalize_answer(reference_text)
    bleu_score = _get_evaluator("bleu").compute([predicted_text], [reference_text])
    rouge_score = _get_evaluator("rouge").compute([predicted_text], [reference_text])
    meteor_score = _get_evaluator("meteor").compute([predicted_text], [reference_text])
    bert_score = _get_evaluator("bertscore").compute([predicted_text], [reference_text])
    token_scores = token_f1_score(predicted_text, reference_text)

    pred_tokens = set(pred_norm.split()) if pred_norm else set()
    ref_tokens = set(ref_norm.split()) if ref_norm else set()
    union = pred_tokens.union(ref_tokens)
    intersection = pred_tokens.intersection(ref_tokens)
    jaccard_similarity = (len(intersection) / len(union)) if union else 0.0

    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)
    length_ratio = (pred_len / ref_len) if ref_len > 0 else 0.0

    exact_match_norm = 1.0 if pred_norm == ref_norm and ref_norm else 0.0

    return {
        "bleu": bleu_score.get("bleu"),
        "rouge_l": rouge_score.get("rouge_l"),
        "meteor": meteor_score.get("meteor"),
        "factual_correctness": bert_score.get("bertscore_f1"),
        "bleu_fallback_used": bleu_score.get("bleu_fallback_used", 0.0),
        "rouge_fallback_used": rouge_score.get("rouge_fallback_used", 0.0),
        "meteor_fallback_used": meteor_score.get("meteor_fallback_used", 0.0),
        "bertscore_fallback_used": bert_score.get("bertscore_fallback_used", 0.0),
        "token_precision": token_scores.get("token_precision"),
        "token_recall": token_scores.get("token_recall"),
        "token_f1": token_scores.get("token_f1"),
        "anls": anls_score(predicted_text, reference_text),
        "exact_match": exact_match_norm,
        "jaccard_similarity": jaccard_similarity,
        "length_ratio": length_ratio,
    }


def _compute_vqa_metrics(predicted_text: str, reference_text: str) -> Dict[str, Optional[float]]:
    return _compute_text_metrics(predicted_text, reference_text)


def evaluate_single_sample(task: str, prediction: PredictionRecord, sample: RadiologySample) -> SampleMetric:
    metric = SampleMetric(
        sample_id=sample.sample_id,
        dataset_name=sample.dataset_name,
        model_name=prediction.model_name,
        task=prediction.task,
    )

    if prediction.error:
        return metric

    if task == "rrg" and sample.report_reference:
        text_metrics = _compute_text_metrics(prediction.predicted_text or "", sample.report_reference)
        chexbert_score = _get_evaluator("chexbert").compute([prediction.predicted_text or ""], [sample.report_reference])
        radgraph_score = _get_evaluator("radgraph").compute([prediction.predicted_text or ""], [sample.report_reference])
        hallucination = _get_evaluator("hallucination").compute([prediction.predicted_text or ""], [sample.report_reference])
        metric.bleu = text_metrics.get("bleu")
        metric.rouge_l = text_metrics.get("rouge_l")
        metric.meteor = text_metrics.get("meteor")
        metric.factual_correctness = text_metrics.get("factual_correctness")
        metric.token_precision = text_metrics.get("token_precision")
        metric.token_recall = text_metrics.get("token_recall")
        metric.token_f1 = text_metrics.get("token_f1")
        metric.anls = text_metrics.get("anls")
        metric.exact_match = text_metrics.get("exact_match")
        metric.jaccard_similarity = text_metrics.get("jaccard_similarity")
        metric.length_ratio = text_metrics.get("length_ratio")
        metric.bleu_fallback_used = text_metrics.get("bleu_fallback_used")
        metric.rouge_fallback_used = text_metrics.get("rouge_fallback_used")
        metric.meteor_fallback_used = text_metrics.get("meteor_fallback_used")
        metric.bertscore_fallback_used = text_metrics.get("bertscore_fallback_used")
        metric.chexbert_f1 = chexbert_score.get("chexbert_f1")
        metric.radgraph_f1 = radgraph_score.get("radgraph_f1")
        metric.chexbert_fallback_used = chexbert_score.get("chexbert_fallback_used")
        metric.radgraph_fallback_used = radgraph_score.get("radgraph_fallback_used")
        metric.hallucination_score = hallucination.get("hallucination_score")

    elif task == "vqa" and sample.answer_reference:
        text_metrics = _compute_vqa_metrics(prediction.predicted_text or "", sample.answer_reference)
        metric.bleu = text_metrics.get("bleu")
        metric.rouge_l = text_metrics.get("rouge_l")
        metric.meteor = text_metrics.get("meteor")
        metric.factual_correctness = text_metrics.get("factual_correctness")
        metric.token_precision = text_metrics.get("token_precision")
        metric.token_recall = text_metrics.get("token_recall")
        metric.token_f1 = text_metrics.get("token_f1")
        metric.anls = text_metrics.get("anls")
        metric.exact_match = text_metrics.get("exact_match")
        metric.jaccard_similarity = text_metrics.get("jaccard_similarity")
        metric.length_ratio = text_metrics.get("length_ratio")
        metric.bleu_fallback_used = text_metrics.get("bleu_fallback_used")
        metric.rouge_fallback_used = text_metrics.get("rouge_fallback_used")
        metric.meteor_fallback_used = text_metrics.get("meteor_fallback_used")
        metric.bertscore_fallback_used = text_metrics.get("bertscore_fallback_used")

    elif task == "grounding" and sample.bounding_boxes and prediction.predicted_bboxes:
        pred_boxes = prediction.predicted_bboxes or []
        ref_boxes = [box.to_dict() for box in sample.bounding_boxes]
        bbox_metrics = _get_evaluator("bbox").compute_set_metrics(pred_boxes, ref_boxes)
        metric.bbox_iou = bbox_metrics.get("mean_iou")
        metric.bbox_precision_025 = bbox_metrics.get("precision@0.25")
        metric.bbox_precision_05 = bbox_metrics.get("precision@0.5")
        metric.bbox_precision_075 = bbox_metrics.get("precision@0.75")
        metric.bbox_recall_05 = bbox_metrics.get("recall@0.5")

        if sample.findings_list:
            reference_text = sample.findings_list[0]
            text_metrics = _compute_text_metrics(prediction.predicted_text or "", reference_text)
            metric.bleu = text_metrics.get("bleu")
            metric.rouge_l = text_metrics.get("rouge_l")
            metric.meteor = text_metrics.get("meteor")
            metric.factual_correctness = text_metrics.get("factual_correctness")
            metric.token_precision = text_metrics.get("token_precision")
            metric.token_recall = text_metrics.get("token_recall")
            metric.token_f1 = text_metrics.get("token_f1")
            metric.anls = text_metrics.get("anls")
            metric.exact_match = text_metrics.get("exact_match")
            metric.jaccard_similarity = text_metrics.get("jaccard_similarity")
            metric.length_ratio = text_metrics.get("length_ratio")
            metric.bleu_fallback_used = text_metrics.get("bleu_fallback_used")
            metric.rouge_fallback_used = text_metrics.get("rouge_fallback_used")
            metric.meteor_fallback_used = text_metrics.get("meteor_fallback_used")
            metric.bertscore_fallback_used = text_metrics.get("bertscore_fallback_used")

    return metric


def build_run_name(preset: str, models: List[str]) -> str:
    compact = "-".join(models[:2])
    suffix = "all" if len(models) <= 2 else f"plus{len(models)-2}"
    return f"{preset}_{compact}_{suffix}".replace("/", "_")


def preflight_model(model_name: str, info: ModelInfo, preset_cfg: Dict[str, Any], backend: str) -> Dict[str, Any]:
    access = check_model_access(model_name)
    if not access["accessible"]:
        return access

    quant = preset_cfg.get("inference", {}).get("quantization")
    mem_gb = preset_cfg.get("hardware", {}).get("max_memory_gb", 0)

    if info.needs_4bit and quant is None and mem_gb <= 16:
        return {
            "accessible": False,
            "requires_key": False,
            "reason": "insufficient_vram",
            "message": (
                f"Model '{model_name}' typically needs 4-bit quantization on <=16GB VRAM. "
                f"Preset quantization is None."
            ),
        }

    if backend == "vllm":
        import importlib.util

        if importlib.util.find_spec("vllm") is None:
            return {
                "accessible": False,
                "requires_key": False,
                "reason": "missing_vllm",
                "message": "vLLM backend requested but package is not installed. Falling back to transformers is recommended.",
            }

    return access


def run_inference(model: Any, task: str, sample: RadiologySample) -> PredictionRecord:
    sample_image = _extract_images(sample)
    pred = PredictionRecord(
        sample_id=sample.sample_id,
        dataset_name=sample.dataset_name,
        model_name=model.model_name,
        task=task,
    )

    try:
        if task == "rrg":
            out = model.generate_report(sample_image)
            pred.predicted_text = out.text
        elif task == "vqa":
            out = model.answer_question(sample_image, sample.question or "")
            pred.predicted_text = out.text
        elif task == "grounding":
            finding = (sample.findings_list[0] if sample.findings_list else "abnormality")
            out = model.ground_finding(sample_image, finding)
            pred.predicted_text = out.text
            pred.predicted_bboxes = out.bounding_boxes
        else:
            raise ValueError(f"Unsupported task: {task}")
    except Exception as exc:
        pred.error = str(exc)
        pred.error_type = type(exc).__name__

    return pred


def compute_paired_stats(metrics: List[Dict[str, Any]], seed: int = 42) -> Dict[str, Any]:
    tester = StatisticalTester(alpha=0.05, seed=seed)
    metric_fields = (
        "bleu",
        "rouge_l",
        "meteor",
        "factual_correctness",
        "token_precision",
        "token_recall",
        "token_f1",
        "anls",
        "exact_match",
        "jaccard_similarity",
        "length_ratio",
        "bleu_fallback_used",
        "rouge_fallback_used",
        "meteor_fallback_used",
        "bertscore_fallback_used",
        "chexbert_fallback_used",
        "radgraph_fallback_used",
        "chexbert_f1",
        "radgraph_f1",
        "hallucination_score",
        "vqa_accuracy",
        "bbox_iou",
        "bbox_precision_025",
        "bbox_precision_05",
        "bbox_precision_075",
        "bbox_recall_05",
    )

    by_task_metric: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in metrics:
        task = row.get("task")
        model = row.get("model_name")
        sample_id = row.get("sample_id")
        if not task or not model or not sample_id:
            continue

        for metric_name in metric_fields:
            value = row.get(metric_name)
            if value is None:
                continue
            key = f"{task}:{metric_name}"
            by_task_metric.setdefault(key, {}).setdefault(model, {})[sample_id] = float(value)

    stats: Dict[str, Any] = {}
    for key, model_rows in by_task_metric.items():
        models = sorted(model_rows.keys())
        if len(models) < 2:
            continue
        stats[key] = {}
        for model_a, model_b in itertools.combinations(models, 2):
            common_ids = sorted(set(model_rows[model_a]).intersection(model_rows[model_b]))
            if len(common_ids) < 3:
                continue
            scores_a = [model_rows[model_a][sid] for sid in common_ids]
            scores_b = [model_rows[model_b][sid] for sid in common_ids]
            stats[key][f"{model_a}__vs__{model_b}"] = tester.compare_models(
                scores_a,
                scores_b,
                model_a_name=model_a,
                model_b_name=model_b,
            )

    return stats


def main() -> None:
    args = parse_args()
    setup_logging(level="DEBUG" if args.debug else "INFO")
    set_global_seed(args.seed)

    manager = EnvironmentManager()
    preset = RuntimePreset(args.preset)
    preset_cfg = manager.get_preset_config(preset)
    env_cfg = manager.build_environment_config(preset)

    models = resolve_models(args.models)
    if not models:
        raise RuntimeError("No models resolved from --models")

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = build_run_name(args.preset, models)

    writer = ResultWriter(output_dir=args.output_dir, run_name=run_name)
    writer.save_config(
        {
            "preset": args.preset,
            "datasets": args.datasets,
            "models": models,
            "num_samples": args.num_samples,
            "backend": args.backend,
            "skip_inaccessible": args.skip_inaccessible,
            "seed": args.seed,
        }
    )
    writer.save_environment(
        {
            "runtime_preset": args.preset,
            "environment_profile": env_cfg.environment.value,
            "preset_hardware": preset_cfg.get("hardware", {}),
            "preset_inference": preset_cfg.get("inference", {}),
            "selected_backend": args.backend,
            "seed": args.seed,
            "git_commit": get_git_commit_hash(),
        }
    )

    preflight_report = {}
    runnable_models: List[str] = []
    for model_name in models:
        info = get_model_info(model_name)
        if info is None:
            preflight_report[model_name] = {
                "accessible": False,
                "reason": "model_not_found",
                "message": "Model does not exist in registry",
            }
            continue
        status = preflight_model(model_name, info, preset_cfg, backend=args.backend)
        preflight_report[model_name] = status

        if status.get("accessible"):
            runnable_models.append(model_name)
        elif not args.skip_inaccessible:
            raise RuntimeError(f"Preflight failed for {model_name}: {status.get('message')}")

    if args.dry_run:
        dry_run_file = Path(args.output_dir) / run_name / "preflight.json"
        with open(dry_run_file, "w") as f:
            json.dump(preflight_report, f, indent=2)
        logger.info("Dry run completed")
        return

    if not runnable_models:
        raise RuntimeError("No runnable models after preflight")

    for dataset_name in args.datasets:
        task, dataset = load_dataset(dataset_name, args.num_samples or env_cfg.num_samples, args.ms_cxr_path)
        logger.info("Dataset=%s task=%s samples=%s", dataset_name, task, len(dataset))

        for model_name in runnable_models:
            logger.info("Running model=%s dataset=%s", model_name, dataset_name)

            # Apply low-VRAM protection from preset defaults.
            runtime_overrides = {
                "max_new_tokens": env_cfg.max_new_tokens,
            }
            if env_cfg.quantization == "4bit":
                runtime_overrides["load_in_4bit"] = True
            if env_cfg.quantization == "8bit":
                runtime_overrides["load_in_8bit"] = True

            model = load_model(model_name, **runtime_overrides)
            if hasattr(model, "load"):
                model.load()

            total_samples = len(dataset)
            for sample_index, raw in enumerate(dataset, start=1):
                sample = normalize_sample(raw, dataset_name=dataset_name, split="test")
                is_valid_sample, validation_error = validate_sample_for_task(sample, task)
                if not is_valid_sample:
                    writer.append_error(
                        {
                            "sample_id": sample.sample_id,
                            "dataset_name": dataset_name,
                            "model_name": model_name,
                            "task": task,
                            "error": validation_error,
                            "error_type": "sample_validation",
                        }
                    )
                    if args.strict_sample_validation:
                        raise RuntimeError(
                            f"Invalid sample for task={task}, sample_id={sample.sample_id}: {validation_error}"
                        )
                    logger.warning(
                        "Skipping invalid sample task=%s sample_id=%s reason=%s",
                        task,
                        sample.sample_id,
                        validation_error,
                    )
                    continue

                sample_preview = (
                    sample.question
                    or sample.report_reference
                    or sample.findings_reference
                    or sample.impression_reference
                    or ""
                )
                sample_preview = " ".join(sample_preview.split())[:120]
                if sample_preview:
                    print(
                        f"Progress model={model_name} dataset={dataset_name} sample={sample_index}/{total_samples} "
                        f"sample_id={sample.sample_id} preview={sample_preview}"
                    )
                else:
                    print(
                        f"Progress model={model_name} dataset={dataset_name} sample={sample_index}/{total_samples} "
                        f"sample_id={sample.sample_id}"
                    )

                pred = run_inference(model, task=task, sample=sample)
                writer.append_prediction(pred)

                metric = evaluate_single_sample(task=task, prediction=pred, sample=sample)
                writer.append_sample_metric(metric)

                if pred.error:
                    writer.append_error(
                        {
                            "sample_id": sample.sample_id,
                            "dataset_name": dataset_name,
                            "model_name": model_name,
                            "task": task,
                            "error": pred.error,
                            "error_type": pred.error_type,
                        }
                    )

                if pred.predicted_text:
                    prediction_preview = " ".join(pred.predicted_text.split())[:120]
                    print(
                        f"Output model={model_name} dataset={dataset_name} sample={sample_index}/{total_samples} "
                        f"prediction={prediction_preview}"
                    )

            del model
            try:
                import gc
                import importlib.util

                gc.collect()
                if importlib.util.find_spec("torch") is not None:
                    torch_mod = __import__("torch")

                    if torch_mod.cuda.is_available():
                        torch_mod.cuda.empty_cache()
            except Exception:
                pass

    aggregate = writer.compute_and_save_aggregate_metrics()
    sample_metrics = writer.load_sample_metrics()
    stats = compute_paired_stats(sample_metrics, seed=args.seed)
    fallback_rates = compute_fallback_rates(sample_metrics)
    if fallback_rates:
        stats["_meta_fallback_rates"] = fallback_rates
    writer.save_statistics(stats)

    if args.max_fallback_rate < 1.0:
        exceeded = {
            key: value
            for key, value in fallback_rates.items()
            if value > args.max_fallback_rate
        }
        if exceeded:
            raise RuntimeError(
                "Fallback rate threshold exceeded: "
                + ", ".join(f"{k}={v:.3f}" for k, v in exceeded.items())
            )

    logger.info("Run completed. Models=%d Aggregate sets=%d", len(runnable_models), len(aggregate))
    logger.info(writer.get_result_summary())


if __name__ == "__main__":
    main()
