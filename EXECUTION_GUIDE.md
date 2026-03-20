# LLM Comparison Radiology Benchmark - Execution Guide

## 🎯 Project Status

**Foundation Layer:** ✅ Complete (This PR)  
**Integration Layer:** ⏳ Team Work (Detailed below)  
**Notebook Layer:** ⏳ Team Work (Refactoring)  
**Testing & Release:** ⏳ Final Phase

---

## 🚀 For Researchers: How to Run

### **1. Colab Free T4 (Fastest Start)**

```bash
# 1. Clone and navigate
git clone https://github.com/Hanketsu3/LLMComparison
cd LLMComparison

# 2. Setup (auto-selects env)
bash setup.sh free_colab_t4

# 3. Test system
jupyter notebook notebooks/00_repo_smoke_test.ipynb

# 4. Run one model (e.g., Qwen2-VL-2B)
jupyter notebook notebooks/models/qwen2_vl_2b.ipynb

# 5. Run full comparison
# (when notebooks are refactored, run:)
# python experiments/run_comparison.py --preset free_colab_t4
```

**Expected output:** `results/run_YYYYMMDD_HHMM/`
```
├── config_snapshot.json
├── predictions.jsonl
├── sample_metrics.jsonl
├── aggregate_metrics.json
└── errors.jsonl
```

---

### **2. Local GPU (RTX 3090+)**

```bash
bash setup.sh gpu_24g
python experiments/run_comparison.py \
    --preset gpu_24g \
    --dataset-subset test \
    --num-samples 200 \
    --output results/
```

**Expected runtime:** ~40 min for 13 models × 200 samples

---

### **3. CPU (Testing Only)**

```bash
bash setup.sh smoke_cpu
jupyter notebook notebooks/00_repo_smoke_test.ipynb
```

**Expected:** Model loads in ~1 min, 2-3 lightweight models tested

---

## 🔧 For Developers: Integration Checklist

### **WEEK 1: Foundation Integration**

#### Task 1.1: Sync `model_registry.py` with `model_taxonomy.py`
**Duration:** 1 hour  
**File:** `src/utils/model_registry.py`

```python
# Add after imports
from src.configs.model_taxonomy import (
    BenchmarkTier, 
    ModelCategory,
    MAIN_BENCHMARK_MODELS,
    EXTRA_TRACK_MODELS,
    ALL_MODELS
)

# Update ModelInfo dataclass
@dataclass
class ModelInfo:
    ...
    benchmark_tier: BenchmarkTier = BenchmarkTier.MAIN  # NEW
    # ...

# Add filter functions
def get_main_benchmark_models() -> List[str]:
    return list(MAIN_BENCHMARK_MODELS.keys())

def get_extra_track_models() -> List[str]:
    return list(EXTRA_TRACK_MODELS.keys())

def get_models_by_benchmark_tier(tier: BenchmarkTier) -> List[ModelInfo]:
    return [m for m in MODEL_REGISTRY.values() if m.benchmark_tier == tier]

# Enhanced error messages
def load_model(name: str):
    info = get_model_info(name)
    if info is None:
        # NEW: Provide taxonomy hint
        from src.configs.model_taxonomy import get_model_taxonomy
        try:
            taxonomy = get_model_taxonomy(name)
            if taxonomy.tier == BenchmarkTier.EXTRA:
                raise ValueError(
                    f"Model '{name}' is in EXTRA TRACK (out-of-scope "
                    f"for main radiology benchmark). Reason: {taxonomy.reason}"
                )
        except:
            pass
        raise ValueError(f"Unknown model: {name}")
    ...
```

**Verification:**
```bash
python -c "
from src.utils.model_registry import get_main_benchmark_models, get_extra_track_models
print(f'Main: {len(get_main_benchmark_models())} models')
print(f'Extra: {len(get_extra_track_models())} models')
assert len(get_main_benchmark_models()) == 13
assert len(get_extra_track_models()) == 8
print('✓ Model registry sync verified')
"
```

---

#### Task 1.2: Reorganize `configs/model_configs/*.yaml`
**Duration:** 30 min  
**Files:** `configs/model_configs/{generalist,domain_adaptive,specialist}.yaml`

**Before:**
```yaml
specialist:
  chexagent: {...}
  got_ocr2: {...}        # ❌ Move to extra
  nougat_base: {...}     # ❌ Move to extra
  matcha_chartqa: {...}  # ❌ Move to extra
```

**After:**
```yaml
# MAIN BENCHMARK
specialist:
  chexagent:
    name: "StanfordAIMI/CheXagent-8b"
    ...
  llava_rad:
    name: "microsoft/llava-rad"
    ...
  radfm:
    name: "StanfordAIMI/RadFM"
    ...

# EXTRA TRACK (OCR / Document)
extra_track_ocr:
  got_ocr2:
    name: "stepfun-ai/GOT-OCR-2.0-hf"
    ...
  nougat_base:
    name: "facebook/nougat-base"
    ...

# EXTRA TRACK (Charts)
extra_track_charts:
  matcha_chartqa:
    name: "google/matcha-chartqa"
    ...
```

**Verification:**
```bash
python -c "
import yaml
with open('configs/model_configs/specialist.yaml') as f:
    data = yaml.safe_load(f)
assert 'specialist' in data
assert 'got_ocr2' not in data.get('specialist', {})
print('✓ Config YAML reorganization verified')
"
```

---

#### Task 1.3: Document Gating Requirements
**Duration:** 30 min  
**File:** `GATING_REQUIREMENTS.md` (new)

```markdown
# Gated/Access-Restricted Models

## Llama-3.2-Vision
- **Status:** Gated on HuggingFace
- **Step 1:** Create HF account (https://huggingface.co/join)
- **Step 2:** Accept terms: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
- **Step 3:** Get API token: https://huggingface.co/settings/tokens
- **Step 4:** Login locally:
  ```bash
  huggingface-cli login
  # Paste token when prompted
  ```
- **Fallback:** Use LLaVA-Med or Phi-3.5 (no gating)

## CheXagent
- **Status:** Requires HF login
- **Same steps as Llama-3.2-Vision**

## GPT-4V, Gemini
- **Status:** Proprietary API
- **Requirement:** OpenAI API key or Google API key
- **Cost:** ~$0.01-0.03 per inference
- **Setup:**
  ```bash
  export OPENAI_API_KEY="sk-..."
  export GOOGLE_API_KEY="..."
  ```
- **Fallback:** Use open-source models (default)

## Checking Access
```python
from src.utils.model_registry import check_model_access

access = check_model_access("llama3-vision")
print(access.is_accessible)  # True/False
print(access.reason)         # "gated" / "missing API key" / "ok"
print(access.fix)            # Instructions to fix
```
```

---

### **WEEK 2: Notebook Refactoring**

#### Task 2.1: Create Foundation Notebooks
**Duration:** 1.5 hours  
**Files:**
- `notebooks/00_repo_smoke_test.ipynb`
- `notebooks/01_dataset_schema_check.ipynb`

```python
# notebooks/00_repo_smoke_test.ipynb - Cell 1
"""
🧪 LLM Comparison - Repo Smoke Test
- Duration: ~5 minutes
- Requirements: None (uses CPU fallback)
- Purpose: Verify installation and imports
"""

# Cell 2
import sys
print(f"Python: {sys.version}")

# Cell 3
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch not installed")

# Cell 4: Import all custom modules
from src.configs.model_taxonomy import MAIN_BENCHMARK_MODELS, EXTRA_TRACK_MODELS
from src.configs.environment import EnvironmentManager, RuntimePreset
from src.data.schema import RadiologySample
from src.backends.base import BackendInterface, TransformersBackend
from src.utils.result_writer import ResultWriter

print(f"✓ {len(MAIN_BENCHMARK_MODELS)} main models, {len(EXTRA_TRACK_MODELS)} extra models")
print("✓ All imports successful")

# Cell 5: Load a tiny model (e.g., SmolVLM2)
from src.utils.model_registry import load_model
model = load_model("smolvlm2-2.2b")
print(f"✓ Model loaded: {model.model_name}")

# Cell 6
print("\n✅ SMOKE TEST PASSED")
```

---

#### Task 2.2: Create Model-Specific Notebooks
**Duration:** 4 hours (8 notebooks × 30 min)

**Template:** `notebooks/models/_template.ipynb`

```python
# Cell 1: Header (Markdown)
"""
# 🏥 Model: {MODEL_NAME}

- **Date:** {DATE}
- **Environment:** {ENV_FILE}
- **Recommended GPU:** {GPU_MIN}
- **Quantization:** {QUANT_STRATEGY}
- **Known Issues:** {LIST}

## Prerequisites
1. Environment: `conda activate llmcomp-{env-name}`
2. Storage: ~{MODEL_SIZE} GB for model weights
3. Gating: {GATING_INFO if gated, else "None"}
"""

# Cell 2: Config
PRESET = "gpu_24g"  # Change per hardware
DATASET_SUBSET = "test"
NUM_SAMPLES = 50
OUTPUT_DIR = "results/"

# Cell 3: Environment validation
from src.configs.environment import EnvironmentManager, RuntimePreset
manager = EnvironmentManager()
config = manager.build_environment_config(RuntimePreset.GPU_24G)
print(f"✓ Config: {config}")

# Cell 4: Dataset loading
from src.data.mimic_cxr import MIMICCXRDataset
dataset = MIMICCXRDataset(
    data_dir="/path/to/mimic_cxr",
    split="test",
    max_samples=NUM_SAMPLES
)
sample = dataset[0]
print(f"✓ Loaded {len(dataset)} samples")
print(f"  Sample schema: {sample.sample_id}, {sample.dataset_name}")

# Cell 5: Model loading
from src.utils.model_registry import load_model
model = load_model("{MODEL_NAME}")
model.load()
print(f"✓ Model loaded: {model.model_name}")

# Cell 6: Inference loop
from src.utils.result_writer import ResultWriter, PredictionRecord
writer = ResultWriter(OUTPUT_DIR, "test_run")

for i, sample in enumerate(dataset):
    try:
        output = model.generate_report(sample.image_path)
        record = PredictionRecord(
            sample_id=sample.sample_id,
            dataset_name=sample.dataset_name,
            model_name=model.model_name,
            task="rrg",
            predicted_text=output.text,
            tokens_generated=len(output.text.split()),
        )
        writer.append_prediction(record)
    except Exception as e:
        print(f"Error on {sample.sample_id}: {e}")

print(f"✓ Predictions saved to {writer.predictions_file}")

# Cell 7: Validation
predictions = writer.load_predictions()
print(f"✓ Saved {len(predictions)} predictions")
```

**Repeat for:** qwen2_vl_2b, phi35_vision, internvl2_2b, llama32_vision, llava_med, chexagent, llava_rad, radfm

---

#### Task 2.3: Create Aggregation Notebook
**Duration:** 1 hour  
**File:** `notebooks/99_result_aggregation.ipynb`

```python
# Cell 1: Load all results
import json
from pathlib import Path
import pandas as pd

results_dir = Path("results")
runs = list(results_dir.glob("**/aggregate_metrics.json"))
print(f"Found {len(runs)} runs")

# Cell 2: Aggregate across runs
all_results = {}
for run_file in runs:
    with open(run_file) as f:
        data = json.load(f)
    all_results[run_file.parent.name] = data["metrics"]

# Cell 3: Create comparison table
df = pd.DataFrame(all_results).T
print(df[["model_name", "bleu_mean", "rouge_l_mean", "radgraph_f1_mean"]].sort_values("radgraph_f1_mean", ascending=False))

# Cell 4: Statistical tests
from scipy import stats
# Paired t-test on sample metrics
for run in runs:
    sample_metrics = run.parent / "sample_metrics.jsonl"
    # Load and compute pairwise tests
    ...

# Cell 5: Plots
import matplotlib.pyplot as plt
# Box plots, correlation heatmaps, etc.
```

---

### **WEEK 3: CLI & Documentation**

#### Task 3.1: Unified Experiment Runner
**Duration:** 1.5 hours  
**File:** `experiments/run_unified.py` (refactor existing)

```python
import argparse
from pathlib import Path
from src.configs.environment import EnvironmentManager, RuntimePreset
from src.configs.model_taxonomy import BenchmarkTier, get_models_by_tier, MAIN_BENCHMARK_MODELS
from src.utils.model_registry import load_model, get_model_info
from src.utils.result_writer import ResultWriter
from src.data.mimic_cxr import MIMICCXRDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="gpu_24g",
                       choices=[p.value for p in RuntimePreset])
    parser.add_argument("--models", nargs="+", default=None,
                       help="Model names, or 'all', 'generalist', 'specialist', 'domain_adaptive'")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--dataset", default="mimic_cxr")
    parser.add_argument("--data-dir", default="${MIMIC_CXR_PATH}")
    
    args = parser.parse_args()
    
    # Setup config
    manager = EnvironmentManager()
    config = manager.build_environment_config(RuntimePreset[args.preset.upper()])
    manager.validate_config(config)
    
    # Select models
    if args.models is None or args.models == ["all"]:
        model_names = list(MAIN_BENCHMARK_MODELS.keys())
    elif args.models == ["generalist"]:
        model_names = [k for k, v in MAIN_BENCHMARK_MODELS.items() if v.category.value == "generalist"]
    else:
        model_names = args.models
    
    print(f"Running {len(model_names)} models on {args.preset}")
    print(f"Models: {model_names}")
    
    # Setup result writer
    from datetime import datetime
    run_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = ResultWriter(args.output, run_name)
    writer.save_config({
        "preset": args.preset,
        "models": model_names,
        "num_samples": args.num_samples,
        "dataset": args.dataset,
    })
    
    # Load dataset
    dataset = MIMICCXRDataset(
        data_dir=args.data_dir,
        split="test",
        max_samples=args.num_samples or config.num_samples
    )
    
    # Run inference
    for model_name in model_names:
        try:
            print(f"\n{'='*60}")
            print(f"Loading {model_name}...")
            model = load_model(model_name)
            model.load()
            
            for i, sample in enumerate(dataset):
                try:
                    output = model.generate_report(sample.image_path)
                    # ... save predictions
                except Exception as e:
                    # ... save error
                    pass
            
            model.unload()
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    
    # Compute aggregates
    writer.compute_and_save_aggregate_metrics()
    print(writer.get_result_summary())

if __name__ == "__main__":
    main()
```

---

#### Task 3.2: Update README.md
**Duration:** 1 hour  
**File:** `README.md`

Add sections:
1. **Benchmark Scope** (table of main vs extra models)
2. **Quick Start by Hardware** (copy from setup.sh output)
3. **Environment Selection** (when to use which env)
4. **Preset Details** (table: hardware, batch size, quant, models)
5. **Gating Requirements** (link to GATING_REQUIREMENTS.md)
6. **Common Errors** (Colab OOM, CUDA issues, etc.)
7. **Reproducibility** (config snapshots, result format)

---

## 📊 Validation Checklist

```bash
# 1. Dependencies
conda create -n test -y
conda activate test
pip install -r requirements.txt
python -c "from src.configs import model_taxonomy; print('✓')"

# 2. Model Registry
python -c "
from src.utils.model_registry import get_main_benchmark_models
assert len(get_main_benchmark_models()) == 13
print('✓ 13 main models')
"

# 3. Data Schema
python -c "
from src.data.schema import RadiologySample
sample = RadiologySample(
    sample_id='test_001',
    dataset_name='test',
    split='test',
    image_path='/tmp/test.jpg',
    report_reference='Test report'
)
assert sample.is_valid_for_task('rrg')
print('✓ Schema validation works')
"

# 4. Result Writer
python -c "
from src.utils.result_writer import ResultWriter
writer = ResultWriter('.', 'test_run')
assert writer.predictions_file.exists()
print('✓ Result writer initialized')
"

# 5. Notebooks
jupyter nbconvert --to notebook --execute notebooks/00_repo_smoke_test.ipynb
echo "✓ Smoke test notebook runs"

# 6. CLI
python experiments/run_unified.py --help
echo "✓ CLI runner works"
```

---

## 📞 Support

If you encounter issues:

1. **Check logs:** `results/run_name/errors.jsonl`
2. **Gating errors:** See `GATING_REQUIREMENTS.md`
3. **Memory errors:** Try smaller preset (e.g., free_colab_t4 → smoke_cpu)
4. **Import errors:** Verify environment: `conda activate llmcomp-{env}`

---

**Status:** Foundation ready ✅ | Next: Integration & Testing  
**Estimated Completion:** 3-4 weeks with full team
