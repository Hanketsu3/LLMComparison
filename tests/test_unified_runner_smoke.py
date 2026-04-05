"""Fast smoke tests for unified runner helpers."""

import pytest

from experiments.run_unified import compute_paired_stats, resolve_models, run_inference
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
        {"task": "vqa", "model_name": "a", "sample_id": "1", "vqa_accuracy": 1.0},
        {"task": "vqa", "model_name": "a", "sample_id": "2", "vqa_accuracy": 1.0},
        {"task": "vqa", "model_name": "a", "sample_id": "3", "vqa_accuracy": 0.0},
        {"task": "vqa", "model_name": "b", "sample_id": "1", "vqa_accuracy": 0.0},
        {"task": "vqa", "model_name": "b", "sample_id": "2", "vqa_accuracy": 0.0},
        {"task": "vqa", "model_name": "b", "sample_id": "3", "vqa_accuracy": 0.0},
    ]
    stats = compute_paired_stats(rows)
    assert "vqa:vqa_accuracy" in stats
