"""
Result Writer - Standardized output format for all experiments

Saves predictions, metrics, stats, and errors in structured JSON/JSONL format
for reproducibility and downstream analysis.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Single prediction record."""
    sample_id: str
    dataset_name: str
    model_name: str
    task: str  # "rrg", "vqa", "grounding"
    
    # Prediction
    predicted_text: Optional[str] = None
    predicted_bboxes: Optional[List[Dict[str, float]]] = None
    
    # Confidence
    confidence: Optional[float] = None
    
    # Generation metadata
    tokens_generated: Optional[int] = None
    latency_ms: Optional[float] = None
    
    # Error if any
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    # Timestamp
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SampleMetric:
    """Single sample's evaluation metrics."""
    sample_id: str
    dataset_name: str
    model_name: str
    task: str
    
    # NLP metrics (for RRG/VQA)
    bleu: Optional[float] = None
    rouge_l: Optional[float] = None
    meteor: Optional[float] = None
    
    # Clinical metrics (radiology-specific)
    radgraph_f1: Optional[float] = None
    chexbert_f1: Optional[float] = None
    
    # VQA metrics
    exact_match: Optional[float] = None
    vqa_accuracy: Optional[float] = None
    
    # Grounding metrics
    bbox_iou: Optional[float] = None
    
    # Hallucination / Safety
    hallucination_score: Optional[float] = None
    factual_correctness: Optional[float] = None
    
    # Metadata
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class ResultWriter:
    """Writes structured experiment results."""
    
    def __init__(self, output_dir: Union[str, Path], run_name: str):
        """
        Initialize result writer.
        
        Args:
            output_dir: Base results directory
            run_name: Name of this run (becomes subdirectory)
        """
        self.output_dir = Path(output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.config_file = self.output_dir / "config_snapshot.json"
        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.sample_metrics_file = self.output_dir / "sample_metrics.jsonl"
        self.aggregate_metrics_file = self.output_dir / "aggregate_metrics.json"
        self.stats_file = self.output_dir / "stats.json"
        self.errors_file = self.output_dir / "errors.jsonl"
        self.environment_file = self.output_dir / "environment.json"
        
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration snapshot."""
        config_to_save = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
        }
        with open(self.config_file, "w") as f:
            json.dump(config_to_save, f, indent=2)
        logger.info(f"Config saved to {self.config_file}")
    
    def save_environment(self, env_info: Dict[str, Any]) -> None:
        """Save environment information."""
        env_to_save = {
            "timestamp": datetime.now().isoformat(),
            "python_version": self._get_python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            **env_info,
        }
        with open(self.environment_file, "w") as f:
            json.dump(env_to_save, f, indent=2)
        logger.info(f"Environment saved to {self.environment_file}")
    
    def append_prediction(self, record: PredictionRecord) -> None:
        """Append single prediction record."""
        record.timestamp = datetime.now().isoformat()
        with open(self.predictions_file, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
    
    def append_predictions(self, records: List[PredictionRecord]) -> None:
        """Append multiple prediction records."""
        for record in records:
            self.append_prediction(record)
    
    def append_sample_metric(self, metric: SampleMetric) -> None:
        """Append single sample metric record."""
        metric.timestamp = datetime.now().isoformat()
        with open(self.sample_metrics_file, "a") as f:
            f.write(json.dumps(metric.to_dict()) + "\n")
    
    def append_sample_metrics(self, metrics: List[SampleMetric]) -> None:
        """Append multiple sample metric records."""
        for metric in metrics:
            self.append_sample_metric(metric)
    
    def append_error(self, record: Dict[str, Any]) -> None:
        """Append error record."""
        record["timestamp"] = datetime.now().isoformat()
        with open(self.errors_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def save_aggregate_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Save aggregate metrics by model.
        
        Args:
            metrics: {model_name: {metric_name: value}}
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        with open(self.aggregate_metrics_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Aggregate metrics saved to {self.aggregate_metrics_file}")
    
    def save_statistics(self, stats: Dict[str, Any]) -> None:
        """Save statistical test results.
        
        Args:
            stats: {test_name: {model_pair: p_value, ...}, ...}
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
        }
        with open(self.stats_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Statistics saved to {self.stats_file}")
    
    def compute_and_save_aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute aggregate metrics from sample_metrics.jsonl."""
        aggregate = {}
        
        if not self.sample_metrics_file.exists():
            logger.warning("No sample metrics file found")
            return aggregate
        
        # Load all sample metrics
        sample_metrics = []
        try:
            with open(self.sample_metrics_file, "r") as f:
                for line in f:
                    if line.strip():
                        sample_metrics.append(json.loads(line))
        except FileNotFoundError:
            return aggregate
        
        # Group by model
        by_model = {}
        for record in sample_metrics:
            model_name = record.get("model_name")
            if model_name not in by_model:
                by_model[model_name] = []
            by_model[model_name].append(record)
        
        # Compute aggregates
        for model_name, records in by_model.items():
            aggregate[model_name] = {}
            
            # Iterate over all possible metrics
            for metric_name in ["bleu", "rouge_l", "meteor", "radgraph_f1", 
                               "chexbert_f1", "exact_match", "vqa_accuracy", "bbox_iou"]:
                values = [r[metric_name] for r in records if metric_name in r and r[metric_name] is not None]
                
                if values:
                    aggregate[model_name][f"{metric_name}_mean"] = float(np.mean(values))
                    aggregate[model_name][f"{metric_name}_std"] = float(np.std(values))
                    aggregate[model_name][f"{metric_name}_min"] = float(np.min(values))
                    aggregate[model_name][f"{metric_name}_max"] = float(np.max(values))
        
        self.save_aggregate_metrics(aggregate)
        return aggregate
    
    def load_sample_metrics(self) -> List[Dict[str, Any]]:
        """Load all sample metrics from JSONL."""
        metrics = []
        if not self.sample_metrics_file.exists():
            return metrics
        
        with open(self.sample_metrics_file, "r") as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        
        return metrics
    
    def load_predictions(self) -> List[Dict[str, Any]]:
        """Load all predictions from JSONL."""
        predictions = []
        if not self.predictions_file.exists():
            return predictions
        
        with open(self.predictions_file, "r") as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        
        return predictions
    
    def get_result_summary(self) -> str:
        """Get summary of results saved."""
        summary = f"\n{'='*80}\n"
        summary += f"📊 RESULTS SAVED TO: {self.output_dir}\n"
        summary += f"{'='*80}\n"
        
        # File statuses
        files = {
            "Config": self.config_file,
            "Environment": self.environment_file,
            "Predictions": self.predictions_file,
            "Sample Metrics": self.sample_metrics_file,
            "Aggregate Metrics": self.aggregate_metrics_file,
            "Statistics": self.stats_file,
            "Errors": self.errors_file,
        }
        
        summary += "\n📁 Output Files:\n"
        for name, path in files.items():
            status = "✓" if path.exists() else "✗"
            summary += f"  {status} {name:20s} → {path.name}\n"
        
        # Quick stats
        try:
            with open(self.aggregate_metrics_file, "r") as f:
                agg = json.load(f).get("metrics", {})
                summary += f"\n📈 Models Evaluated: {len(agg)}\n"
        except:
            pass
        
        summary += f"\n{'='*80}\n"
        return summary
    
    @staticmethod
    def _get_python_version() -> str:
        """Get Python version string."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
