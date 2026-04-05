"""Tests for evaluation modules.

Test Coverage:
- Clinical metrics (RadGraph F1)
- NLP metrics (BLEU, ROUGE)
- Grounding evaluation (Bounding box IoU)
- Statistical testing utilities
"""

import pytest
import importlib.util


@pytest.mark.evaluation
@pytest.mark.unit
class TestRadGraphF1:
    """Test RadGraph F1 evaluator."""
    
    def test_radgraph_init(self):
        from src.evaluation.clinical_metrics.radgraph_f1 import RadGraphF1Evaluator
        
        evaluator = RadGraphF1Evaluator(reward_level="partial")
        assert evaluator.name == "radgraph_f1"

    @pytest.mark.unit
    def test_radgraph_compute_with_stubbed_graph(self, monkeypatch):
        from src.evaluation.clinical_metrics.radgraph_f1 import RadGraphF1Evaluator

        evaluator = RadGraphF1Evaluator(reward_level="full")

        def _fake_extract_graph(text):
            if "normal" in text:
                return (
                    {("heart size", "OBSERVATION")},
                    {("heart size", "located_at", "mediastinum")},
                )
            return (
                {("pleural effusion", "OBSERVATION")},
                {("pleural effusion", "located_at", "left base")},
            )

        monkeypatch.setattr(evaluator, "_extract_graph", _fake_extract_graph)
        out = evaluator.compute(["heart size normal"], ["heart size normal"])

        assert out["radgraph_f1"] == pytest.approx(1.0)
        assert "radgraph_entity_f1" in out
        assert "radgraph_relation_f1" in out


class TestCheXbertF1:
    """Test CheXbert evaluator fallback behavior."""

    @pytest.mark.unit
    def test_chexbert_fallback_scores_nonzero_on_match(self):
        from src.evaluation.clinical_metrics.chexbert_f1 import CheXbertF1Evaluator

        evaluator = CheXbertF1Evaluator()
        out = evaluator.compute(
            ["Mild cardiomegaly and pleural effusion are present."],
            ["There is cardiomegaly with pleural effusion."],
        )

        assert "chexbert_f1" in out
        assert out["chexbert_f1"] > 0.0

    @pytest.mark.unit
    def test_chexbert_negation_handling(self):
        from src.evaluation.clinical_metrics.chexbert_f1 import CheXbertF1Evaluator

        evaluator = CheXbertF1Evaluator()
        out = evaluator.compute(["No pleural effusion."], ["Pleural effusion is present."])
        assert out["chexbert_micro_f1"] < 1.0


class TestFactCheXcker:
    """Test hallucination evaluator outputs."""

    @pytest.mark.unit
    def test_factchexcker_returns_factual_metrics(self):
        from src.evaluation.hallucination.factchexcker import FactCheXckerEvaluator

        evaluator = FactCheXckerEvaluator()
        out = evaluator.compute(
            ["There is pleural effusion and edema."],
            ["Pleural effusion is present."],
        )

        assert "hallucination_score" in out
        assert "factual_f1" in out
        assert out["total_hallucinations"] >= 0


class TestBLEU:
    """Test BLEU evaluator."""
    
    def test_bleu_compute(self):
        if importlib.util.find_spec("evaluate") is None and importlib.util.find_spec("nltk") is None:
            pytest.skip("BLEU dependencies (evaluate or nltk) are not installed in this environment.")

        from src.evaluation.nlp_metrics.bleu import BLEUEvaluator
        
        evaluator = BLEUEvaluator()
        
        predictions = ["The heart is normal size"]
        references = ["The heart is normal size"]
        
        results = evaluator.compute(predictions, references)
        
        assert "bleu" in results
        assert results["bleu"] > 0


class TestBBoxEvaluator:
    """Test bounding box evaluator."""
    
    def test_iou_computation(self):
        from src.evaluation.grounding.bbox_metrics import BBoxEvaluator
        
        evaluator = BBoxEvaluator()
        
        # Perfect overlap
        pred = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        ref = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        
        iou = evaluator._compute_iou(pred, ref)
        assert iou == 1.0
    
    def test_iou_no_overlap(self):
        from src.evaluation.grounding.bbox_metrics import BBoxEvaluator
        
        evaluator = BBoxEvaluator()
        
        pred = {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        ref = {"x_min": 20, "y_min": 20, "x_max": 30, "y_max": 30}
        
        iou = evaluator._compute_iou(pred, ref)
        assert iou == 0.0

    def test_compute_set_metrics_multi_bbox(self):
        from src.evaluation.grounding.bbox_metrics import BBoxEvaluator

        evaluator = BBoxEvaluator()
        preds = [
            {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
            {"x_min": 20, "y_min": 20, "x_max": 30, "y_max": 30},
        ]
        refs = [
            {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
            {"x_min": 100, "y_min": 100, "x_max": 110, "y_max": 110},
        ]

        out = evaluator.compute_set_metrics(preds, refs)
        assert "mean_iou" in out
        assert "precision@0.5" in out
        assert "recall@0.5" in out
        assert out["precision@0.5"] == pytest.approx(0.5)
        assert out["recall@0.5"] == pytest.approx(0.5)


class TestStatisticalTester:
    """Test statistical testing utilities."""
    
    def test_paired_t_test(self):
        if importlib.util.find_spec("scipy") is None:
            pytest.skip("scipy is not installed in this environment.")

        from src.utils.statistical_tests import StatisticalTester
        
        tester = StatisticalTester(alpha=0.05)
        
        scores_a = [0.8, 0.85, 0.82, 0.88, 0.84]
        scores_b = [0.7, 0.72, 0.68, 0.75, 0.71]
        
        result = tester.paired_t_test(scores_a, scores_b)
        
        assert "p_value" in result
        assert "statistic" in result
        assert result["p_value"] < 0.05  # Should be significant
    
    def test_bootstrap_ci(self):
        from src.utils.statistical_tests import StatisticalTester
        
        tester = StatisticalTester(seed=7)
        
        scores = [0.8, 0.85, 0.82, 0.88, 0.84]
        lower, upper = tester.bootstrap_confidence_interval(scores)
        
        assert lower < upper
        assert lower > 0.7
        assert upper < 0.95

    def test_bootstrap_ci_is_deterministic_with_seed(self):
        from src.utils.statistical_tests import StatisticalTester

        scores = [0.8, 0.85, 0.82, 0.88, 0.84]
        tester_a = StatisticalTester(seed=17)
        tester_b = StatisticalTester(seed=17)

        ci_a = tester_a.bootstrap_confidence_interval(scores, n_bootstrap=200)
        ci_b = tester_b.bootstrap_confidence_interval(scores, n_bootstrap=200)

        assert ci_a == ci_b
