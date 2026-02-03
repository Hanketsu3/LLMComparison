"""
Statistical Testing Utilities

For significance testing between model comparisons.
"""

from typing import Dict, List, Tuple


# Note: numpy imported inside methods for lazy loading


class StatisticalTester:
    """Statistical significance testing for model comparisons."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def paired_t_test(
        self,
        scores_a: List[float],
        scores_b: List[float]
    ) -> Dict[str, float]:
        """Perform paired t-test."""
        from scipy import stats
        
        statistic, p_value = stats.ttest_rel(scores_a, scores_b)
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
        }
    
    def wilcoxon_test(
        self,
        scores_a: List[float],
        scores_b: List[float]
    ) -> Dict[str, float]:
        """Perform Wilcoxon signed-rank test."""
        from scipy import stats
        
        try:
            statistic, p_value = stats.wilcoxon(scores_a, scores_b)
        except ValueError:
            # All differences are zero
            return {"statistic": 0, "p_value": 1.0, "significant": False}
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
        }
    
    def bootstrap_confidence_interval(
        self,
        scores: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        import numpy as np
        
        scores = np.array(scores)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
        
        return float(lower), float(upper)
    
    def compare_models(
        self,
        model_a_scores: List[float],
        model_b_scores: List[float],
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> Dict:
        """Comprehensive comparison between two models."""
        import numpy as np
        
        results = {
            "comparison": f"{model_a_name} vs {model_b_name}",
            "model_a_mean": float(np.mean(model_a_scores)),
            "model_b_mean": float(np.mean(model_b_scores)),
            "difference": float(np.mean(model_a_scores) - np.mean(model_b_scores)),
            "paired_t_test": self.paired_t_test(model_a_scores, model_b_scores),
            "wilcoxon_test": self.wilcoxon_test(model_a_scores, model_b_scores),
        }
        
        # Add confidence intervals
        a_ci = self.bootstrap_confidence_interval(model_a_scores)
        b_ci = self.bootstrap_confidence_interval(model_b_scores)
        results["model_a_ci"] = a_ci
        results["model_b_ci"] = b_ci
        
        return results
