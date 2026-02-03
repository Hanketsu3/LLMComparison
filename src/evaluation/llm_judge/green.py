"""
GREEN (GPT-4 Radiology Error Evaluation) Metric

Uses GPT-4 as a judge to evaluate clinical errors in generated reports.
"""

import logging
import os
from typing import Dict, List, Optional
from src.evaluation.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class GREENEvaluator(BaseEvaluator):
    """GREEN metric using GPT-4 as an expert judge for radiology reports."""
    
    EVALUATION_PROMPT = """You are an expert radiologist evaluating AI-generated radiology reports.

Compare the generated report to the reference report and identify:
1. False Positives: Findings mentioned in generated but NOT in reference
2. False Negatives: Findings in reference but MISSING from generated
3. Incorrect Details: Wrong anatomical locations, severity, or descriptions

Reference Report:
{reference}

Generated Report:
{prediction}

Rate the generated report on a scale of 1-5:
1 = Major clinical errors that could harm patient
2 = Significant errors affecting diagnosis
3 = Minor errors, overall acceptable
4 = Minor omissions, clinically acceptable
5 = Excellent match, no significant errors

Respond with JSON:
{{"score": <1-5>, "false_positives": [list], "false_negatives": [list], "errors": [list]}}"""

    def __init__(
        self,
        judge_model: str = "gpt-4",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name="green", **kwargs)
        self.judge_model = judge_model
        self.api_key = api_key
        self.client = None
    
    def _load_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install: pip install openai")
        
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute GREEN score using GPT-4."""
        if self.client is None:
            self._load_client()
        
        scores = []
        fp_counts = []
        fn_counts = []
        
        for pred, ref in zip(predictions, references):
            try:
                result = self._evaluate_single(pred, ref)
                scores.append(result.get("score", 3))
                fp_counts.append(len(result.get("false_positives", [])))
                fn_counts.append(len(result.get("false_negatives", [])))
            except Exception as e:
                logger.warning(f"GREEN evaluation failed: {e}")
                scores.append(3)  # Neutral score
        
        return {
            "green_score": sum(scores) / len(scores) if scores else 0,
            "green_score_normalized": (sum(scores) / len(scores) - 1) / 4 if scores else 0,
            "avg_false_positives": sum(fp_counts) / len(fp_counts) if fp_counts else 0,
            "avg_false_negatives": sum(fn_counts) / len(fn_counts) if fn_counts else 0,
        }
    
    def _evaluate_single(self, prediction: str, reference: str) -> Dict:
        """Evaluate a single prediction-reference pair."""
        import json
        
        prompt = self.EVALUATION_PROMPT.format(
            reference=reference,
            prediction=prediction
        )
        
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON response
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        return {"score": 3}
