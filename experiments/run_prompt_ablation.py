"""
Prompt Ablation Experiment Runner

Compare different prompt strategies on the same model and dataset.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import yaml
from tqdm import tqdm

from src.utils import setup_logging, PromptManager
from src.evaluation.clinical_metrics import RadGraphF1Evaluator
from src.evaluation.nlp_metrics import BLEUEvaluator, ROUGEEvaluator

logger = logging.getLogger(__name__)


# Model registry for easy instantiation
MODEL_REGISTRY = {
    # Generalist (Free)
    "minicpm-v": ("src.models.generalist.minicpm_v", "MiniCPMVModel"),
    "qwen2-vl": ("src.models.generalist.qwen2_vl", "Qwen2VLModel"),
    "internvl2": ("src.models.generalist.internvl2", "InternVL2Model"),
    "llava-next": ("src.models.generalist.llava_next", "LLaVANextModel"),
    "phi3-vision": ("src.models.generalist.phi3_vision", "Phi3VisionModel"),
    "llama3": ("src.models.generalist.llama3", "Llama3Model"),
    # Generalist (API - costs money)
    "gpt4v": ("src.models.generalist.gpt4v", "GPT4VModel"),
    "gemini": ("src.models.generalist.gemini", "GeminiModel"),
    # Domain-Adaptive (Free)
    "llava-med": ("src.models.domain_adaptive.llava_med", "LLaVAMedModel"),
    "biomedgpt": ("src.models.domain_adaptive.biomedgpt", "BiomedGPTModel"),
    # Specialist (Free)
    "chexagent": ("src.models.specialist.chexagent", "CheXagentModel"),
    "llava-rad": ("src.models.specialist.llava_rad", "LLaVARadModel"),
    "radfm": ("src.models.specialist.radfm", "RadFMModel"),
}


def get_model(model_name: str):
    """Dynamically load and instantiate a model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    module_path, class_name = MODEL_REGISTRY[model_name]
    
    import importlib
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    return model_class()


def run_prompt_ablation(
    model_name: str,
    dataset,
    prompts: List[str],
    num_samples: int = 50
) -> Dict[str, Dict[str, float]]:
    """Run ablation study over different prompts."""
    
    logger.info(f"Running prompt ablation for {model_name} with {len(prompts)} prompts")
    
    # Load model
    model = get_model(model_name)
    model.load()
    
    # Initialize prompt manager and evaluators
    prompt_manager = PromptManager()
    evaluators = [RadGraphF1Evaluator(), BLEUEvaluator(), ROUGEEvaluator()]
    
    # Get samples
    samples = list(dataset)[:num_samples]
    references = [s.full_report for s in samples]
    
    results = {}
    
    for prompt_name in prompts:
        logger.info(f"Testing prompt: {prompt_name}")
        
        try:
            prompt_template = prompt_manager.get_prompt("rrg", prompt_name)
        except ValueError:
            logger.warning(f"Unknown prompt: {prompt_name}, skipping")
            continue
        
        predictions = []
        
        for sample in tqdm(samples, desc=f"Prompt: {prompt_name}"):
            try:
                formatted = prompt_template.format()
                output = model.generate_report(
                    sample.image_path, 
                    prompt=formatted["user"]
                )
                predictions.append(output.text)
            except Exception as e:
                logger.error(f"Error: {e}")
                predictions.append("")
        
        # Evaluate
        scores = {}
        for evaluator in evaluators:
            try:
                result = evaluator.compute(predictions, references)
                scores.update(result)
            except Exception as e:
                logger.error(f"Evaluation error with {evaluator.name}: {e}")
        
        results[prompt_name] = scores
        logger.info(f"{prompt_name}: {scores}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Prompt ablation experiment")
    parser.add_argument("--model", type=str, default="qwen2-vl",
                        help="Model to test (default: qwen2-vl)")
    parser.add_argument("--prompts", type=str, nargs="+", 
                        default=["baseline", "detailed", "structured", "chain_of_thought"],
                        help="Prompt variants to test")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/prompt_ablation.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    setup_logging(level="DEBUG" if args.debug else "INFO")
    logger.info(f"Prompt ablation: {args.model} with prompts {args.prompts}")
    
    # Load dataset (placeholder)
    from src.data import MIMICCXRDataset
    dataset = MIMICCXRDataset(data_dir="./data/mimic-cxr", split="test")
    
    # Run ablation
    results = run_prompt_ablation(
        model_name=args.model,
        dataset=dataset,
        prompts=args.prompts,
        num_samples=args.num_samples
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            "model": args.model,
            "prompts_tested": args.prompts,
            "num_samples": args.num_samples,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PROMPT ABLATION RESULTS")
    print("="*60)
    print(f"Model: {args.model}")
    print("-"*60)
    
    for prompt_name, scores in results.items():
        print(f"\n{prompt_name}:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
