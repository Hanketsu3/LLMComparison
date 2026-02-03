#!/usr/bin/env python
"""
Demo script - Test project imports and basic functionality.

Run this to verify everything is set up correctly (no dataset needed).
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    
    errors = []
    
    # Core utils
    try:
        from src.utils import PromptManager, print_model_table
        print("  ✅ src.utils")
    except ImportError as e:
        errors.append(f"  ❌ src.utils: {e}")
    
    # Model registry
    try:
        from src.utils.model_registry import FREE_MODELS, PAID_MODELS, COLAB_MODELS
        print(f"  ✅ Model registry ({len(FREE_MODELS)} free, {len(PAID_MODELS)} paid)")
    except ImportError as e:
        errors.append(f"  ❌ model_registry: {e}")
    
    # Evaluation
    try:
        from src.evaluation.nlp_metrics import BLEUEvaluator, ROUGEEvaluator
        print("  ✅ src.evaluation.nlp_metrics")
    except ImportError as e:
        errors.append(f"  ❌ evaluation: {e}")
    
    # Base model
    try:
        from src.models.base_model import BaseRadiologyModel, ModelOutput
        print("  ✅ src.models.base_model")
    except ImportError as e:
        errors.append(f"  ❌ base_model: {e}")
    
    if errors:
        print("\n⚠️ Import errors:")
        for err in errors:
            print(err)
        return False
    
    print("\n✅ All imports successful!")
    return True


def test_prompt_manager():
    """Test prompt manager functionality."""
    print("\nTesting PromptManager...")
    
    from src.utils import PromptManager
    
    pm = PromptManager()
    prompts = pm.list_prompts()
    
    print(f"  Found {len(prompts.get('rrg', []))} RRG prompts")
    
    for name in ["baseline", "detailed", "turkish"]:
        try:
            p = pm.get_prompt("rrg", name)
            print(f"  ✅ {name}: {len(p.user_prompt)} chars")
        except ValueError:
            print(f"  ⚠️ {name}: not found")
    
    print("✅ PromptManager works!")


def test_model_registry():
    """Test model registry."""
    print("\nModel Registry:")
    
    from src.utils import print_model_table
    print_model_table()


def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM Comparison - Project Demo")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Fix import errors before continuing.")
        return 1
    
    # Test prompt manager
    test_prompt_manager()
    
    # Show model registry
    test_model_registry()
    
    print("\n" + "=" * 60)
    print("✅ Project is ready to use!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Set up environment variables (copy .env.example to .env)")
    print("  2. Download dataset (see docs)")
    print("  3. Run: python experiments/run_comparison.py --list-models")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
