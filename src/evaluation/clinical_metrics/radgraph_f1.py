"""RadGraph evaluator with entity and relation-level scoring."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

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
        self._model_is_available = False
    
    def _load_model(self) -> None:
        """Load RadGraph model."""
        try:
            from radgraph import RadGraph  # type: ignore
            self.model = RadGraph()
            self._model_is_available = True
            logger.info("Loaded RadGraph model")
        except Exception as exc:
            self.model = None
            self._model_is_available = False
            logger.warning("RadGraph package unavailable (%s). Using fallback extractor.", exc)
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute RadGraph F1 score."""
        if self.model is None and not self._model_is_available:
            self._load_model()
        
        entity_scores: List[Tuple[float, float, float]] = []
        relation_scores: List[Tuple[float, float, float]] = []
        blended_scores: List[Tuple[float, float, float]] = []

        for pred, ref in zip(predictions, references):
            try:
                pred_entities, pred_relations = self._extract_graph(pred)
                ref_entities, ref_relations = self._extract_graph(ref)

                if not ref_entities and not ref_relations:
                    continue

                ent_prf = self._compute_prf(pred_entities, ref_entities)
                rel_prf = self._compute_prf(pred_relations, ref_relations)

                entity_scores.append(ent_prf)
                relation_scores.append(rel_prf)

                if self.reward_level == "full":
                    blended_scores.append(
                        (
                            0.5 * (ent_prf[0] + rel_prf[0]),
                            0.5 * (ent_prf[1] + rel_prf[1]),
                            0.5 * (ent_prf[2] + rel_prf[2]),
                        )
                    )
                else:
                    blended_scores.append(ent_prf)
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue

        if not blended_scores:
            return {
                "radgraph_precision": 0.0,
                "radgraph_recall": 0.0,
                "radgraph_f1": 0.0,
                "radgraph_entity_f1": 0.0,
                "radgraph_relation_f1": 0.0,
                "radgraph_fallback_used": 0.0 if self._model_is_available else 1.0,
            }

        blend_p, blend_r, blend_f1 = self._mean_prf(blended_scores)
        _, _, entity_f1 = self._mean_prf(entity_scores)
        _, _, relation_f1 = self._mean_prf(relation_scores)

        return {
            "radgraph_precision": float(blend_p),
            "radgraph_recall": float(blend_r),
            "radgraph_f1": float(blend_f1),
            "radgraph_entity_f1": float(entity_f1),
            "radgraph_relation_f1": float(relation_f1),
            "radgraph_fallback_used": 0.0 if self._model_is_available else 1.0,
        }

    @staticmethod
    def _compute_prf(pred_set: Set[Any], ref_set: Set[Any]) -> Tuple[float, float, float]:
        if not pred_set and not ref_set:
            return 1.0, 1.0, 1.0
        if not ref_set:
            return 0.0, 0.0, 0.0

        matches = len(pred_set & ref_set)
        precision = matches / len(pred_set) if pred_set else 0.0
        recall = matches / len(ref_set) if ref_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    @staticmethod
    def _mean_prf(rows: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        if not rows:
            return 0.0, 0.0, 0.0
        n = float(len(rows))
        return (
            sum(r[0] for r in rows) / n,
            sum(r[1] for r in rows) / n,
            sum(r[2] for r in rows) / n,
        )

    def _extract_graph(self, text: str) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]]]:
        """Extract entities and relations from text using RadGraph."""
        if not text.strip():
            return set(), set()

        if not self._model_is_available or self.model is None:
            return self._extract_graph_fallback(text)

        try:
            result = self.model([text])
            payload = result[0] if isinstance(result, list) and result else result
            entities_payload = payload.get("entities", {}) if isinstance(payload, dict) else {}

            entities: Set[Tuple[str, str]] = set()
            entity_map: Dict[str, Tuple[str, str]] = {}

            for entity_id, entity_data in entities_payload.items():
                tokens = entity_data.get("tokens", "")
                if isinstance(tokens, list):
                    token_text = " ".join(str(t) for t in tokens)
                else:
                    token_text = str(tokens)

                label = str(entity_data.get("label", ""))
                normalized_entity = (token_text.strip().lower(), label)
                entities.add(normalized_entity)
                entity_map[str(entity_id)] = normalized_entity

            relations: Set[Tuple[str, str, str]] = set()
            for entity_id, entity_data in entities_payload.items():
                head = entity_map.get(str(entity_id))
                if head is None:
                    continue
                for rel in entity_data.get("relations", []) or []:
                    rel_label = ""
                    target_id = None
                    if isinstance(rel, (list, tuple)) and len(rel) >= 2:
                        rel_label, target_id = rel[0], rel[1]
                    elif isinstance(rel, dict):
                        rel_label = rel.get("label") or rel.get("relation") or ""
                        target_id = rel.get("target") or rel.get("target_id")

                    target = entity_map.get(str(target_id)) if target_id is not None else None
                    if target is None:
                        continue
                    relations.add((head[0], str(rel_label), target[0]))

            return entities, relations
        except Exception as e:
            logger.warning(f"RadGraph extraction failed: {e}")
            return self._extract_graph_fallback(text)

    def _extract_graph_fallback(self, text: str) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]]]:
        """Fallback extractor using clinically common entity phrases."""
        patterns = {
            "OBSERVATION": [
                "cardiomegaly",
                "pleural effusion",
                "pneumothorax",
                "consolidation",
                "atelectasis",
                "edema",
                "opacity",
            ],
            "ANATOMY": ["left base", "right base", "lung", "heart", "mediastinum", "pleura"],
        }

        text_lower = text.lower()
        entities: Set[Tuple[str, str]] = set()
        for label, terms in patterns.items():
            for term in terms:
                if term in text_lower:
                    entities.add((term, label))

        relations: Set[Tuple[str, str, str]] = set()
        anatomy = [e[0] for e in entities if e[1] == "ANATOMY"]
        observations = [e[0] for e in entities if e[1] == "OBSERVATION"]
        for obs in observations:
            for anat in anatomy:
                relations.add((obs, "located_at", anat))

        return entities, relations
