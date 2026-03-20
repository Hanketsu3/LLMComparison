#!/usr/bin/env python3
"""
Quick validation script to test that fixed models can be loaded and run inference.
This script tests each model with 1-2 samples to verify basic functionality.
"""

import os
import sys
import logging
from pathlib import Path
from PIL import Image
import torch
import gc

# Setup paths
if os.path.exists("src"):
    sys.path.insert(0, os.getcwd())

from src.utils.model_registry import MODEL_REGISTRY, load_model
from src.data.hf_vqa_rad import HFVQARADDataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Test models (should match lane-based lists in the notebook)
MODELS_TO_TEST = [
    # Generalist
    "qwen3-vl-2b",
    "qwen2.5-vl-3b",
    "qwen2-vl-2b",
    "phi3-vision",
    "smolvlm2-2.2b",
    "llama3-vision",
    # Domain-adaptive
    "qwen2-vl-ocr-2b",
    "latxa-qwen3-vl-2b",
    "medgemma-4b",
    # Specialist
    "got-ocr2",
    "nougat-base",
    "matcha-chartqa",
]

def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def test_single_model(model_name: str, test_image, test_question) -> bool:
    """Test a single model with one sample."""
    try:
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}")
        
        # Load model info
        info = MODEL_REGISTRY.get(model_name)
        if not info:
            print(f"❌ Model {model_name} not found in registry")
            return False
        
        print(f"   Display Name: {info.display_name}")
        print(f"   Category: {info.category.value}")
        print(f"   Params: {info.params}")
        
        # Try to load model
        print(f"   ⏳ Loading model...")
        model = load_model(model_name)
        model.load()
        print(f"   ✅ Model loaded successfully")
        
        # Try to run inference
        print(f"   ⏳ Running inference...")
        
        # Save image temporarily
        tmp_path = "/tmp/test_img.png"
        test_image.save(tmp_path)
        
        try:
            output = model.answer_question(tmp_path, test_question)
            answer = output.text.strip()
            
            # Check that we got some output
            if answer and len(answer) > 0:
                print(f"   ✅ Inference successful")
                print(f"   Answer: {answer[:100]}...")
                return True
            else:
                print(f"   ❌ Model returned empty answer")
                return False
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            del model
            cleanup_gpu()
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)[:200]}")
        cleanup_gpu()
        return False

def main():
    print("\n" + "="*70)
    print("🧪 MODEL VALIDATION TEST")
    print("="*70)
    print(f"Testing {len(MODELS_TO_TEST)} models with quick validation...")
    
    # Load a test image and question
    try:
        print(f"\n📊 Loading VQA-RAD dataset (first 2 samples)...")
        dataset = HFVQARADDataset(split="test", max_samples=2)
        sample = dataset[0]
        test_image = sample['image']
        test_question = sample['question']
        print(f"   Sample question: {test_question}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print(f"   Using dummy image instead...")
        test_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        test_question = "Is there any abnormality visible?"
    
    # Test each model
    results = {}
    for model_name in MODELS_TO_TEST:
        success = test_single_model(model_name, test_image, test_question)
        results[model_name] = "✅ PASS" if success else "❌ FAIL"
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for v in results.values() if "✅" in v)
    total = len(results)
    
    for model_name, status in results.items():
        print(f"{status} | {model_name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} models passed validation")
    print(f"{'='*70}")
    
    if passed == total:
        print("\n🎉 All models passed! Ready to run full experiment.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} model(s) failed. Check above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
