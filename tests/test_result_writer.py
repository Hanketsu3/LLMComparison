"""Tests for structured results writer output format."""

import json

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
            vqa_accuracy=1.0,
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


@pytest.mark.integration
def test_result_writer_stats_file(tmp_path):
    writer = ResultWriter(output_dir=tmp_path, run_name="stats_run")
    writer.save_statistics({"vqa:vqa_accuracy": {"a__vs__b": {"p": 0.01}}})

    stats_path = tmp_path / "stats_run" / "stats.json"
    with open(stats_path, "r") as f:
        payload = json.load(f)
    assert "statistics" in payload
