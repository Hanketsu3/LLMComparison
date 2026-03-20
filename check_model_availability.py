#!/usr/bin/env python3
"""
Quick model availability check - verifies fixed models exist on HuggingFace.
This is a lightweight pre-flight check before running full experiment.
"""

import sys
from huggingface_hub import model_info, ModelNotFound

# Models to verify (lane-based shortlist in run_full_experiment notebook)
MODELS_TO_CHECK = {
    # Generalist lane
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "phi3-vision": "microsoft/Phi-3.5-vision-instruct",
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "smolvlm2-2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "llama3-vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    # Domain-adaptive lane
    "qwen2-vl-ocr-2b": "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
    "latxa-qwen3-vl-2b": "HiTZ/Latxa-Qwen3-VL-2B-Instruct",
    "medgemma-4b": "google/medgemma-4b-it",
    # Specialist lane
    "got-ocr2": "stepfun-ai/GOT-OCR-2.0-hf",
    "nougat-base": "facebook/nougat-base",
    "matcha-chartqa": "google/matcha-chartqa",
}

def check_model_availability():
    """Check if all models exist on HuggingFace."""
    print("\n" + "="*70)
    print("🔍 MODEL AVAILABILITY CHECK")
    print("="*70)
    print("Checking if all models exist on HuggingFace...\n")
    
    results = {}
    
    for local_name, hf_model_id in MODELS_TO_CHECK.items():
        try:
            info = model_info(hf_model_id)
            results[local_name] = {
                "status": "✅ Found",
                "model_id": hf_model_id,
                "private": info.private,
                "gated": getattr(info, "gated", False),
            }
            print(f"✅ {local_name:<20} → {hf_model_id}")
            
        except ModelNotFound:
            results[local_name] = {
                "status": "❌ NOT FOUND",
                "model_id": hf_model_id,
            }
            print(f"❌ {local_name:<20} → {hf_model_id} (NOT FOUND)")
            
        except Exception as e:
            results[local_name] = {
                "status": "⚠️  ERROR",
                "model_id": hf_model_id,
                "error": str(e)[:100],
            }
            error_msg = str(e)[:50]
            if "401" in error_msg or "gated" in error_msg.lower():
                print(f"⚠️  {local_name:<20} → {hf_model_id} (GATED REPO)")
            else:
                print(f"⚠️  {local_name:<20} → ERROR: {error_msg}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    found = sum(1 for r in results.values() if "❌" not in r["status"])
    not_found = sum(1 for r in results.values() if "❌" in r["status"])
    errors = sum(1 for r in results.values() if "⚠️" in r["status"])
    
    print(f"\n✅ Found:      {found}")
    print(f"❌ Not Found:  {not_found}")
    print(f"⚠️  Gated/Error: {errors}")
    
    if not_found == 0:
        print("\n🎉 All core models are available! You can proceed with the experiment.")
        if errors > 0:
            print("\n📝 Note: Gated/private repos need HF login during notebook run.")
        return 0
    else:
        print("\n⚠️  Some models are not available. Check the errors above.")
        if not_found > 0:
            print("   These models may have been deleted or moved on HuggingFace.")
        return 1

if __name__ == "__main__":
    exit_code = check_model_availability()
    sys.exit(exit_code)
