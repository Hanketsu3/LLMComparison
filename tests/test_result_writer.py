"""Tests for structured results writer output format."""

import json
import math

import pytest

from src.utils.result_writer import PredictionRecord, ResultWriter, SampleMetric


@pytest.mark.unit
def test_result_writer_creates_expected_files(tmp_path):
    writer = ResultWriter(output_dir=tmp_path, run_name="unit_run")

    writer.save_config({"task": "vqa"})
    writer.save_environment({"runtime": "cpu"})

    writer.append_prediction(
        PredictionRecord(
            sample_id="1",
            dataset_name="hf_vqa_rad",
            model_name="dummy",
            task="vqa",
            predicted_text="yes",
        )
    )
    writer.append_sample_metric(
        SampleMetric(
            sample_id="1",
            dataset_name="hf_vqa_rad",
            model_name="dummy",
            task="vqa",
            token_f1=1.0,
            anls=0.75,
            bleu_fallback_used=0.0,
            hallucination_score=0.25,
        )
    )

    aggregate = writer.compute_and_save_aggregate_metrics()

    run_dir = tmp_path / "unit_run"
    assert (run_dir / "config_snapshot.json").exists()
    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "sample_metrics.jsonl").exists()
    assert (run_dir / "aggregate_metrics.json").exists()
    assert aggregate["dummy"]["hallucination_score_mean"] == 0.25
    assert aggregate["dummy"]["token_f1_mean"] == 1.0
    assert aggregate["dummy"]["anls_mean"] == 0.75
    assert aggregate["dummy"]["bleu_fallback_used_mean"] == 0.0


@pytest.mark.unit
def test_result_writer_ignores_nan_metrics(tmp_path):
    writer = ResultWriter(output_dir=tmp_path, run_name="nan_run")
    writer.append_sample_metric(
        SampleMetric(
            sample_id="1",
            dataset_name="hf_vqa_rad",
            model_name="dummy",
            task="vqa",
            token_f1=1.0,
            anls=math.nan,
        )
    )

    aggregate = writer.compute_and_save_aggregate_metrics()
    assert aggregate["dummy"]["token_f1_mean"] == 1.0
    assert "anls_mean" not in aggregate["dummy"]


@pytest.mark.unit
def test_result_writer_grounding_metrics_aggregate(tmp_path):
    writer = ResultWriter(output_dir=tmp_path, run_name="grounding_run")
    writer.append_sample_metric(
        SampleMetric(
            sample_id="1",
            dataset_name="ms_cxr",
            model_name="dummy",
            task="grounding",
            bbox_iou=0.6,
            bbox_precision_025=1.0,
            bbox_precision_05=1.0,
            bbox_precision_075=0.0,
            bbox_recall_05=0.5,
        )
    )
    aggregate = writer.compute_and_save_aggregate_metrics()
    assert aggregate["dummy"]["bbox_iou_mean"] == 0.6
    assert aggregate["dummy"]["bbox_precision_05_mean"] == 1.0
    assert aggregate["dummy"]["bbox_recall_05_mean"] == 0.5


@pytest.mark.integration
def test_result_writer_stats_file(tmp_path):
    writer = ResultWriter(output_dir=tmp_path, run_name="stats_run")
    writer.save_statistics({"vqa:token_f1": {"a__vs__b": {"p": 0.01}}})

    stats_path = tmp_path / "stats_run" / "stats.json"
    with open(stats_path, "r") as f:
        payload = json.load(f)
    assert "statistics" in payload
