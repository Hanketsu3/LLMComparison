"""Fast smoke tests for unified runner helpers."""

import pytest

from experiments.run_unified import (
    _normalize_question_type,
    compute_fallback_rates,
    compute_paired_stats,
    resolve_models,
    run_inference,
    validate_sample_for_task,
)
from src.data.schema import RadiologySample


class _DummyOutput:
    def __init__(self, text):
        self.text = text
        self.bounding_boxes = [{"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}]


class _DummyModel:
    model_name = "dummy-model"

    def generate_report(self, image):
        return _DummyOutput("findings normal")

    def answer_question(self, image, question):
        return _DummyOutput("yes")

    def ground_finding(self, image, finding):
        return _DummyOutput("box")


@pytest.mark.unit
def test_resolve_models_groups():
    models = resolve_models(["generalist", "specialist"])
    assert "qwen2-vl-2b" in models
    assert "chexagent" in models


@pytest.mark.unit
def test_run_inference_vqa_dummy():
    model = _DummyModel()
    sample = RadiologySample(
        sample_id="1",
        dataset_name="hf_vqa_rad",
        split="test",
        image_path="/tmp/fake.png",
        question="Is heart enlarged?",
        answer_reference="yes",
    )
    pred = run_inference(model=model, task="vqa", sample=sample)
    assert pred.predicted_text == "yes"
    assert pred.error is None


@pytest.mark.unit
def test_compute_paired_stats_minimal():
    rows = [
        {"task": "vqa", "model_name": "a", "sample_id": "1", "token_f1": 1.0, "anls": 1.0},
        {"task": "vqa", "model_name": "a", "sample_id": "2", "token_f1": 1.0, "anls": 1.0},
        {"task": "vqa", "model_name": "a", "sample_id": "3", "token_f1": 0.0, "anls": 0.0},
        {"task": "vqa", "model_name": "b", "sample_id": "1", "token_f1": 0.0, "anls": 0.0},
        {"task": "vqa", "model_name": "b", "sample_id": "2", "token_f1": 0.0, "anls": 0.0},
        {"task": "vqa", "model_name": "b", "sample_id": "3", "token_f1": 0.0, "anls": 0.0},
    ]
    stats = compute_paired_stats(rows)
    assert "vqa:token_f1" in stats
    assert "vqa:anls" in stats


@pytest.mark.unit
def test_normalize_question_type():
    assert _normalize_question_type("yes/no") == "closed"
    assert _normalize_question_type("Open-Ended") == "open"
    assert _normalize_question_type("COUNT") == "counting"


@pytest.mark.unit
def test_validate_sample_for_task_vqa_invalid_missing_answer():
    sample = RadiologySample(
        sample_id="1",
        dataset_name="hf_vqa_rad",
        split="test",
        image_path="/tmp/fake.png",
        question="Is there edema?",
        answer_reference=None,
    )
    ok, reason = validate_sample_for_task(sample, "vqa")
    assert not ok
    assert reason is not None


@pytest.mark.unit
def test_compute_fallback_rates_summary():
    rows = [
        {"bleu_fallback_used": 0.0, "meteor_fallback_used": 1.0},
        {"bleu_fallback_used": 1.0, "meteor_fallback_used": 1.0},
    ]
    rates = compute_fallback_rates(rows)
    assert rates["bleu_fallback_used"] == pytest.approx(0.5)
    assert rates["meteor_fallback_used"] == pytest.approx(1.0)


@pytest.mark.unit
def test_validate_sample_for_task_grounding_invalid_bbox():
    sample = RadiologySample(
        sample_id="g1",
        dataset_name="ms_cxr",
        split="test",
        image_path="/tmp/fake.png",
        bounding_boxes=[],
    )
    ok, reason = validate_sample_for_task(sample, "grounding")
    assert not ok
    assert reason is not None
