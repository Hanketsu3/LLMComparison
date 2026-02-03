"""
Run Comparison Experiment

Main script for comparing models on Report Generation (RRG), VQA, and Grounding.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import yaml
from tqdm import tqdm

from src.data import (
    MIMICCXRDataset, 
    VQARADDataset, 
    MSCXRDataset
)
from src.evaluation.clinical_metrics import RadGraphF1Evaluator
from src.evaluation.hallucination import FactCheXckerEvaluator
from src.evaluation.grounding import BBoxEvaluator
from src.evaluation.nlp_metrics import BLEUEvaluator
from src.evaluation.vqa_metrics import VQAAccuracyEvaluator
from src.utils.statistical_tests import StatisticalTester
from src.utils.logging import setup_logging
from src.utils.model_registry import (
    load_model, 
    print_model_table, 
    FREE_MODELS, 
    PAID_MODELS
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_model(model, dataset, task: str = "report_generation", num_samples: int = None) -> List[Dict]:
    """Run a model on the dataset and collect predictions."""
    predictions = []
    
    samples = list(dataset)
    if num_samples:
        samples = samples[:num_samples]
    
    for sample in tqdm(samples, desc=f"Running {model.model_name} on {task}"):
        try:
            if task == "report_generation":
                output = model.generate_report(sample.image_path)
                predictions.append({"text": output.text, "ref": getattr(sample, "report", "")})
            elif task == "visual_question_answering":
                output = model.answer_question(sample.image_path, sample.question)
                predictions.append({"text": output.text, "ref": getattr(sample, "answer", "")})
            elif task == "grounding":
                output = model.ground_finding(sample.image_path, sample.finding)
                predictions.append({
                    "bboxes": output.bounding_boxes, 
                    "ref_bboxes": getattr(sample, "bounding_boxes", [])
                })
        except Exception as e:
            logger.error(f"Error processing sample {getattr(sample, 'study_id', 'unknown')}: {e}")
            predictions.append({})
    
    return predictions


def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    evaluators: List
) -> Dict[str, float]:
    """Evaluate predictions with multiple metrics."""
    results = {}
    
    for evaluator in evaluators:
        try:
            scores = evaluator.compute(predictions, references)
            results.update(scores)
        except Exception as e:
            logger.error(f"Error in {evaluator.name}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run model comparison experiment")
    parser.add_argument("--config", type=str, default="configs/experiment_configs/rrg_experiment.yaml")
    parser.add_argument("--output", type=str, default="results/comparison_results.json")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models to test. Use 'free', 'paid', or specific model names")
    parser.add_argument("--free-only", action="store_true", help="Only test free models")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.debug else "INFO")
    
    # List models and exit
    if args.list_models:
        print_model_table()
        return
    
    # Determine which models to test
    if args.models:
        if args.models == ["free"]:
            models_to_test = FREE_MODELS
            logger.info("Testing FREE models only (no API costs)")
        elif args.models == ["paid"]:
            models_to_test = PAID_MODELS
            logger.info("Testing PAID models (API costs apply)")
        else:
            models_to_test = args.models
    elif args.free_only:
        models_to_test = FREE_MODELS
        logger.info("Testing FREE models only (no API costs)")
    else:
        # Default: free models only
        models_to_test = FREE_MODELS
        logger.info("Testing FREE models (use --models paid for paid models)")
    
    logger.info(f"Models to test: {models_to_test}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    # Determine task from config
    task = config.get("experiment", {}).get("task", "report_generation")
    logger.info(f"Task: {task}")

    # Load dataset (Placeholder - typically would use a factory)
    # Load dataset
    logger.info("Loading dataset...")
    data_path = config.get("dataset", {}).get("path")
    
    if task == "report_generation":
        dataset = MIMICCXRDataset(
            data_dir=data_path or "./data/mimic-cxr",
            split="test",
            max_samples=args.num_samples,
        )
    elif task == "visual_question_answering":
        dataset = VQARADDataset(
            data_dir=data_path or "./data/vqa-rad",
            split="test",
            max_samples=args.num_samples,
        )
    elif task == "grounding":
        dataset = MSCXRDataset(
            data_dir=data_path or "./data/ms-cxr",
            split="test",
            max_samples=args.num_samples,
        )
    else:
        logger.warning(f"Unknown task {task}, using dummy list.")
        dataset = []

    # Initialize evaluators based on task
    evaluators = []
    if task == "report_generation":
        evaluators = [
            RadGraphF1Evaluator(),
            FactCheXckerEvaluator(),
        ]
    elif task == "visual_question_answering":
        evaluators = [
            VQAAccuracyEvaluator(), 
            BLEUEvaluator()
        ]
    elif task == "grounding":
        evaluators = [
            BBoxEvaluator()
        ]
    
    # Run experiments
    results = {}
    all_predictions = {}
    
    for model_name in models_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {model_name}...")
        logger.info(f"{'='*60}")
        
        try:
            model = load_model(model_name)
            model.load()
            
            predictions = run_model(model, dataset, task=task, num_samples=args.num_samples)
            all_predictions[model_name] = predictions
            
            # Extract lists for evaluation
            preds_text = [p.get("text", "") for p in predictions]
            refs_text = [p.get("ref", "") for p in predictions]
            
            if evaluators and preds_text and refs_text:
                scores = evaluate_predictions(preds_text, refs_text, evaluators)
                results[model_name] = scores
                logger.info(f"{model_name} results: {scores}")
            else:
                results[model_name] = {"status": "completed", "count": len(predictions)}
                logger.info(f"{model_name}: {len(predictions)} predictions generated.")
            
            # Clean up GPU memory
            del model
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error with {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    # Statistical comparison
    logger.info("Performing statistical comparisons...")
    tester = StatisticalTester(alpha=0.05)
    
    comparisons = []
    model_names = [m for m in results.keys() if "error" not in results[m]]
    
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            scores_a = [results[model_a].get("radgraph_f1", 0)] * len(references)
            scores_b = [results[model_b].get("radgraph_f1", 0)] * len(references)
            
            comparison = tester.compare_models(scores_a, scores_b, model_a, model_b)
            comparisons.append(comparison)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        "config": args.config,
        "num_samples": args.num_samples,
        "models_tested": models_to_test,
        "free_models_used": [m for m in models_to_test if m in FREE_MODELS],
        "paid_models_used": [m for m in models_to_test if m in PAID_MODELS],
        "model_results": results,
        "statistical_comparisons": comparisons,
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print("\n🆓 FREE MODELS:")
    for model_name in [m for m in results.keys() if m in FREE_MODELS]:
        scores = results[model_name]
        if "error" not in scores:
            print(f"  {model_name}:")
            for metric, value in scores.items():
                print(f"    {metric}: {value:.4f}")
    
    if any(m in PAID_MODELS for m in results.keys()):
        print("\n💰 PAID MODELS:")
        for model_name in [m for m in results.keys() if m in PAID_MODELS]:
            scores = results[model_name]
            if "error" not in scores:
                print(f"  {model_name}:")
                for metric, value in scores.items():
                    print(f"    {metric}: {value:.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
