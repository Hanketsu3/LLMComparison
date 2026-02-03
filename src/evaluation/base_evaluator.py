"""
Base Evaluator class for all evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEvaluator(ABC):
    """Abstract base class for evaluation metrics."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
    
    @abstractmethod
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute the metric for given predictions and references."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
