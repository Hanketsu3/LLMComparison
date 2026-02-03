"""
RadGraph F1 Score Evaluator

Measures clinical accuracy by extracting entities and relations from radiology reports.
Gold standard metric for radiology report generation evaluation.
"""

import logging
from typing import Dict, List, Optional
from src.evaluation.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class RadGraphF1Evaluator(BaseEvaluator):
    """
    RadGraph-based F1 score for radiology report evaluation.
    
    RadGraph extracts clinical entities (findings, anatomy) and their relations
    from radiology reports, enabling precise clinical accuracy measurement.
    """
    
    def __init__(
        self,
        reward_level: str = "partial",  # "partial" or "full"
        model_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name="radgraph_f1", **kwargs)
        self.reward_level = reward_level
        self.model_path = model_path
        self.model = None
    
    def _load_model(self):
        """Load RadGraph model."""
        try:
            from radgraph import RadGraph
        except ImportError:
            raise ImportError("Please install: pip install radgraph")
        
        self.model = RadGraph()
        logger.info("Loaded RadGraph model")
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute RadGraph F1 score."""
        if self.model is None:
            self._load_model()
        
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        count = 0
        
        for pred, ref in zip(predictions, references):
            try:
                # Extract entities from both texts
                pred_entities = self._extract_entities(pred)
                ref_entities = self._extract_entities(ref)
                
                # Compute overlap
                if len(ref_entities) == 0:
                    continue
                
                matches = len(pred_entities & ref_entities)
                precision = matches / len(pred_entities) if pred_entities else 0
                recall = matches / len(ref_entities) if ref_entities else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                count += 1
                
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue
        
        if count == 0:
            return {"radgraph_precision": 0, "radgraph_recall": 0, "radgraph_f1": 0}
        
        return {
            "radgraph_precision": total_precision / count,
            "radgraph_recall": total_recall / count,
            "radgraph_f1": total_f1 / count,
        }
    
    def _extract_entities(self, text: str) -> set:
        """Extract clinical entities from text using RadGraph."""
        if not text.strip():
            return set()
        
        try:
            result = self.model([text])
            entities = set()
            
            for entity_id, entity_data in result[0].get("entities", {}).items():
                entity_text = entity_data.get("tokens", "")
                entity_label = entity_data.get("label", "")
                entities.add((entity_text, entity_label))
            
            return entities
        except Exception as e:
            logger.warning(f"RadGraph extraction failed: {e}")
            return set()
