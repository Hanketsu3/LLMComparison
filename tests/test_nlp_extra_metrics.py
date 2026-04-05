"""Tests for additional NLP metrics evaluators."""

import pytest

from src.evaluation.nlp_metrics.bertscore import BERTScoreEvaluator
from src.evaluation.nlp_metrics.meteor import METEOREvaluator


@pytest.mark.unit
def test_meteor_evaluator_runs():
    evaluator = METEOREvaluator()
    out = evaluator.compute(["heart size normal"], ["heart size normal"])
    assert "meteor" in out


@pytest.mark.unit
def test_bertscore_evaluator_runs():
    evaluator = BERTScoreEvaluator()
    out = evaluator.compute(["no pleural effusion"], ["there is no pleural effusion"])
    assert "bertscore_f1" in out
