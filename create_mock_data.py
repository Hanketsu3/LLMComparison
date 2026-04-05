#!/usr/bin/env python3
"""Create mock results directory with sample data for testing."""

import json
from pathlib import Path
import numpy as np

np.random.seed(42)

# Create mock results structure
results_dir = Path("results/demo_test_run")
results_dir.mkdir(parents=True, exist_ok=True)

# Mock sample metrics
models = ["qwen2-vl-2b", "phi-3.5-vision", "internvl2-2b"]
datasets = ["hf_vqa_rad", "hf_grounding_rad"]
tasks = ["vqa", "finding", "grounding"]

sample_metrics = []
for sample_id in range(50):
    model = np.random.choice(models)
    dataset = np.random.choice(datasets)
    task = np.random.choice(tasks)
    
    # Realistic metric values
    bleu = np.random.normal(0.65, 0.10)
    rouge_l = np.random.normal(0.68, 0.12)
    f1_score = np.random.normal(0.70, 0.08)
    
    sample_metrics.append({
        "sample_id": f"sample_{sample_id:04d}",
        "model_name": model,
        "dataset_name": dataset,
        "task": task,
        "bleu": float(max(0, min(1, bleu))),
        "rouge_l": float(max(0, min(1, rouge_l))),
        "f1_score": float(max(0, min(1, f1_score))),
        "bleu_fallback_used": bool(False),
        "rouge_l_fallback_used": bool(np.random.choice([True, False], p=[0.05, 0.95])),
        "f1_fallback_used": bool(False),
    })

# Write sample_metrics.jsonl
sample_metrics_file = results_dir / "sample_metrics.jsonl"
sample_metrics_file.write_text("\n".join(json.dumps(s) for s in sample_metrics))
print(f"✓ Created: {sample_metrics_file} ({len(sample_metrics)} samples)")

# Create config snapshot
config = {
    "timestamp": "2026-04-05T10:00:00",
    "seed": 42,
    "models": models,
    "datasets": datasets,
    "num_samples": 50,
}
config_file = results_dir / "config_snapshot.json"
config_file.write_text(json.dumps(config, indent=2))
print(f"✓ Created: {config_file}")

# Create stats
stats = {
    "statistics": {
        "total_samples": len(sample_metrics),
        "_meta_fallback_rates": {
            "all_evaluators_fallback_rate": 0.038,
        }
    }
}
stats_file = results_dir / "stats.json"
stats_file.write_text(json.dumps(stats, indent=2))
print(f"✓ Created: {stats_file}")

print(f"\n✅ Mock data created in: {results_dir}")
print(f"   Ready for notebook execution")
