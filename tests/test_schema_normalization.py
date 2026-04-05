"""Tests for unified sample schema normalization."""

import pytest

from experiments.run_unified import normalize_sample


@pytest.mark.unit
def test_normalize_vqa_sample_dict():
    raw = {
        "study_id": "123",
        "question": "Is there pleural effusion?",
        "answer": "no",
        "image_path": "/tmp/fake.png",
    }
    sample = normalize_sample(raw, dataset_name="hf_vqa_rad", split="test")

    assert sample.sample_id == "123"
    assert sample.dataset_name == "hf_vqa_rad"
    assert sample.question == "Is there pleural effusion?"
    assert sample.answer_reference == "no"


@pytest.mark.unit
def test_normalize_grounding_boxes():
    raw = {
        "sample_id": "abc",
        "image_path": "/tmp/fake.png",
        "bounding_boxes": [{"x_min": 1, "y_min": 2, "x_max": 10, "y_max": 12, "label": "opacity"}],
    }
    sample = normalize_sample(raw, dataset_name="ms_cxr", split="test")

    assert sample.sample_id == "abc"
    assert sample.bounding_boxes is not None
    assert len(sample.bounding_boxes) == 1
    assert sample.bounding_boxes[0].x_max == 10
