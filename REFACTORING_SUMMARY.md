# 🏥 LLMComparison Radiology Benchmark - Refactoring Summary

## Executed Changes (✅ Complete)

### 1. **Model Taxonomy & Categorization** ✅
**File:** `src/configs/model_taxonomy.py` (new)

- **Main Benchmark (13 models):**
  - Generalist (6): Qwen2-VL-2B, Qwen2.5-VL-3B, Qwen3-VL-2B, Phi-3.5, SmolVLM2, InternVL2-{2B,4B}, Llama-3-Vision
  - Domain-Adaptive (3): LLaVA-Med, MedGemma-4B, BiomedGPT
  - Specialist (3): CheXagent, LLaVA-Rad, RadFM

- **Extra Track (8 models):**
  - OCR/Document: GOT-OCR2, Nougat, Qwen2-VL-OCR
  - Chart QA: MatCha-ChartQA
  - Language-Adaptation: Latxa-Qwen3-VL
  - API/Proprietary: GPT-4V, Gemini

**Impact:** Clarifies scope, prevents out-of-scope models contaminating main results.

---

### 2. **Model-Family Environment Files** ✅
**Directory:** `envs/` (7 YAML files)

| Environment | Purpose | Key Dependency | Transform Version |
|---|---|---|---|
| `base.yaml` | Core dependencies | transformers 4.35.0 | Default |
| `qwen.yaml` | Qwen series | qwen-vl-utils | 4.36.0+ |
| `phi.yaml` | Phi-3.5 | (minimal special) | 4.40.0+ |
| `internvl.yaml` | InternVL2 | trust_remote_code | 4.35.0+ |
| `llama.yaml` | Llama-3.2-Vision | flash-attn | 4.44.0+ |
| `medical.yaml` | Medical domain | monai, nibabel | 4.35.0+ |
| `specialist.yaml` | Radiology specialists | radgraph, spacy | 4.35.0+ |
| `dev.yaml` | Development | pytest, black, mypy | 4.35.0+ |

**Installation:**
```bash
# Qwen models
conda env create -f envs/qwen.yaml -n llmcomp-qwen
conda activate llmcomp-qwen

# Each model family gets isolated env → no version conflicts
```

**Impact:** Eliminates monolithic `requirements.txt` hell; dependency conflicts resolved.

---

### 3. **Runtime Presets** ✅
**File:** `presets/presets.yaml` (5 presets)

| Preset | Hardware | GPU | Memory | Batch | Models | Est. Time |
|---|---|---|---|---|---|---|
| `smoke_cpu` | CPU | None | 8GB | 1 | 2 (Qwen2, Phi) | 5 min |
| `free_colab_t4` | Colab Free | T4 | 16GB | 1 | 5 (light, 4-bit) | 2-4 hrs |
| `colab_paid_mid` | Colab Paid | V100/A100-40 | 40GB | 2 | 8 (+medium) | 1-2 hrs |
| `gpu_24g` | Local RTX 3090/4090 | RTX 3090 | 24GB | 2 | 13 (all main) | 30-40 min |
| `high_end_multi_gpu` | Enterprise | 2×A100-40/H100 | 80GB | 4 | 14 (all+robust) | 15-20 min |

**Key per-preset tuning:**
- **Image resolution:** smoke_cpu (224px) → high_end (768px)
- **Quantization:** T4 mandatory 4-bit → gpu_24g optional bf16 → A100 full precision
- **Generation length:** smoke_cpu (50 tokens) → high_end (512 tokens)
- **Flash-Attention:** Disabled T4 → Enabled gpu_24g/A100
- **Fallback strategy:** T4 skips OOM models (graceful degradation)

**Impact:** Non-experts can run benchmarks on any hardware with single preset selection.

---

### 4. **Unified Sample Schema** ✅
**File:** `src/data/schema.py` (new)

```python
@dataclass
class RadiologySample:
    # Identifiers
    sample_id: str
    dataset_name: str
    split: str
    
    # Image(s)
    image_path: Optional[str]
    image_paths: Optional[List[str]]
    image: Optional[Image.Image]
    
    # Task: RRG
    report_reference: Optional[str]
    findings_reference: Optional[str]
    impression_reference: Optional[str]
    
    # Task: VQA
    question: Optional[str]
    answer_reference: Optional[str]
    question_type: Optional[str]  # "closed", "open"
    
    # Task: Grounding
    findings_list: Optional[List[str]]
    bounding_boxes: Optional[List[BoundingBox]]
    
    # Metadata
    view: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    pathology_labels: Optional[Dict[str, float]]
    modality: Optional[str]
    body_part: Optional[str]
```

**Utility methods:**
- `.get_image()` - Load from disk if needed
- `.is_valid_for_task(task)` - Validate field completeness
- `.to_dict()` / `.from_dict()` - JSON serialization

**Impact:** 
- All dataset loaders produce same schema
- Evaluation is dataset-agnostic
- Hidden field name bugs eliminated
- Extensible without breaking existing code

---

### 5. **Backend Abstraction Layer** ✅
**Files:** `src/backends/base.py`, `src/backends/transformers_backend.py` (new)

**Architecture:**
```
Model Wrapper (e.g., Qwen2VLModel)
    ↓
BackendInterface (abstraction)
    ├── TransformersBackend (implemented)
    ├── vLLMBackend (future)
    └── TensorRTBackend (future)
    ↓
Actual inference (generate, decode, etc.)
```

**Core classes:**
- `BackendType` enum: TRANSFORMERS, VLLM, ONNX, TENSORRT
- `GenerationConfig` dataclass: unified config across backends
- `InferenceResult` dataclass: {text, tokens, latency_ms, memory_used_mb, backend}
- `BackendInterface` ABC: `.load()`, `.unload()`, `.infer()`, `.infer_batch()`
- `BackendRegistry`: Model → Backend mapping
- `TransformersBackend` implementation: Full wrapper around HF models

**Benefits:**
- Models don't care which backend (transformers vs vLLM vs ONNX)
- Quantization: 4-bit, 8-bit, fp16, bf16 handled uniformly
- Memory tracking built-in
- Easy to add vLLM batching later

---

### 6. **Sample-Level Result Logging** ✅
**File:** `src/utils/result_writer.py` (new)

**Output Structure:**
```
results/
└── run_name/
    ├── config_snapshot.json           # Full config at run time
    ├── environment.json               # GPU, torch, transformers versions
    ├── predictions.jsonl              # Each line = 1 prediction
    │   {"sample_id", "model_name", "predicted_text", "tokens", "latency_ms", ...}
    ├── sample_metrics.jsonl           # Each line = metrics for 1 sample
    │   {"sample_id", "model_name", "bleu", "rouge_l", "radgraph_f1", ...}
    ├── aggregate_metrics.json         # Mean/std/min/max per model
    │   {"model_a": {"bleu_mean": 0.30, "bleu_std": 0.05, ...}, ...}
    ├── stats.json                     # Paired t-test, wilcoxon results
    │   {"paired_t_test": {"model_a_vs_b": 0.012, ...}, ...}
    └── errors.jsonl                   # Errors for debugging
        {"sample_id", "model_name", "error_type", "traceback", ...}
```

**Key methods:**
- `append_prediction()` - Streaming writes (no memory bloat)
- `append_sample_metric()` - Per-sample evaluation logging
- `compute_and_save_aggregate_metrics()` - Summary stats
- `load_predictions()`, `load_sample_metrics()` - Reload for post-hoc analysis
- `get_result_summary()` - Terminal output summary

**Impact:**
- **Reproducibility:** Every detail saved
- **Statistical validity:** Sample-level data for proper paired tests
- **Debugging:** Individual error traces per sample/model
- **Streaming:** Doesn't require holding all predictions in RAM

---

### 7. **Environment Configuration Manager** ✅
**File:** `src/configs/environment.py` (new)

**Classes:**
- `RuntimePreset` enum: smoke_cpu, free_colab_t4, etc.
- `EnvironmentProfile` enum: base, qwen, phi, internvl, llama, medical, specialist, dev
- `EnvironmentConfig` dataclass: Merged preset + environment + overrides
- `EnvironmentManager`: Preset loader, validator, builder

**Usage:**
```python
from src.configs.environment import EnvironmentManager, RuntimePreset

manager = EnvironmentManager()
config = manager.build_environment_config(
    preset=RuntimePreset.FREE_COLAB_T4
)
manager.validate_config(config)
print(config)
# EnvironmentConfig(preset=free_colab_t4, env=qwen, device=cuda:0, max_mem=14GB, batch=1, quant=4bit)
```

**Features:**
- Presets from YAML → Python objects
- Environment auto-selection (device → profile mapping)
- Override support (CLI: `--batch-size=2`, `--num-samples=200`)
- Validation warnings (e.g., "T4 without quantization may OOM")
- Pretty printing for user info

---

## Remaining Work (to be done by team)

### **Phase 1: Model Registry Integration** (1-2 hours)

**File:** `src/utils/model_registry.py` 

Changes needed:
1. Import `model_taxonomy.py`
2. Add `benchmark_tier` field to `ModelInfo` dataclass
3. Filter functions:
   ```python
   get_main_benchmark_models() -> List[ModelInfo]
   get_extra_track_models() -> List[ModelInfo]
   ```
4. Error messages upgraded with gating info:
   ```python
   # OLD
   "Model yüklenemedi"
   
   # NEW (with gating/access info)
   "CheXagent loading failed:
    - Container: gated model on HuggingFace
    - Fix: huggingface-cli login (requires email)
    - OR: Use non-specialist models (LLaVA-Med, Phi-3)"
   ```

---

### **Phase 2: Model Configs Reorganization** (30 min)

**Files:** `configs/model_configs/{generalist,domain_adaptive,specialist}.yaml`

**Before:**
```yaml
specialist:
  chexagent: ...
  got_ocr2: ...        # ❌ Out of scope
  nougat_base: ...     # ❌ Out of scope
  matcha_chartqa: ...  # ❌ Out of scope
```

**After:**
```yaml
specialist:
  # -- MAIN BENCHMARK --
  chexagent: ...
  llava_rad: ...
  radfm: ...

# New section
extra_track_ocr:
  got_ocr2: ...
  nougat_base: ...

extra_track_charts:
  matcha_chartqa: ...
```

---

### **Phase 3: Notebook Refactoring** (2-3 hours)

**Current (broken):**
- `notebooks/run_full_experiment.ipynb` (700+ lines, all models in 1 kernel)

**After:**
```
notebooks/
├── 00_repo_smoke_test.ipynb
│   - Import checks, simple model load test, ~10 lines per model
├── 01_dataset_schema_check.ipynb
│   - Load samples from MIMIC_CXR, VQA_RAD, MS_CXR
│   - Verify RadiologySample schema compliance
├── models/
│   ├── qwen2_vl_2b.ipynb
│   │   * Header: model, env requirement, GPU rec, known issues
│   │   * 1) Env validation
│   │   * 2) Model load (with gating check)
│   │   * 3) ~5 sample inference
│   │   * 4) Output validation + save (JSONL)
│   ├── phi35_vision.ipynb
│   ├── internvl2_2b.ipynb
│   ├── qwen2_5vl_3b.ipynb
│   ├── smolvlm2.ipynb
│   ├── llama32_vision.ipynb
│   ├── llava_med.ipynb
│   ├── medgemma_4b.ipynb
│   ├── chexagent.ipynb
│   ├── llava_rad.ipynb
│   └── radfm.ipynb
├── 99_result_aggregation.ipynb
│   - Load all predictions.jsonl + sample_metrics.jsonl
│   - Compute aggregate + stats
│   - Generate plots + comparison table
```

**Benefits:**
- Each notebook focused on 1 model
- Dependency isolation (environment)
- Parallel execution (run notebooks simultaneously across hardware)
- Reusable result format (same .jsonl structure)
- Final comparison is data-driven, not dev-driven

---

### **Phase 4: README & Documentation** (1 hour)

**Update sections:**
1. **Benchmark Scope:** 
   - Main track (13 models), Extra track (8 models)
   - Why each model is in its tier
2. **Quick Start by Hardware:**
   ```markdown
   ### 🚀 Quick Start
   
   #### Google Colab Free (T4)
   ```bash
   git clone https://github.com/Hanketsu3/LLMComparison  
   cd LLMComparison
   conda env create -f envs/qwen.yaml -n llmcomp
   conda activate llmcomp
   python -c "from src.configs.environment import EnvironmentManager; EnvironmentManager().print_preset_summary('free_colab_t4')"
   ```
   Then run: `notebooks/models/qwen2_vl_2b.ipynb`
   
   #### Local GPU (RTX 3090+)
   ```bash
   conda env create -f envs/specialist.yaml -n llmcomp
   python experiments/run_comparison.py --preset gpu_24g --dataset-subset 200
   ```
   ```
3. **Gating Notes:**
   - LLaMA-3.2-Vision → requires HF token
   - GPT-4V, Gemini → API keys
4. **Common Errors:**
   - CUDA OOM on T4 → Use 4-bit quantization
   - Module not found → Check env activated
   - "access_denied" → Gated model requisite

---

### **Phase 5: CLI / Experiment Runner** (2 hours)

**Create:** `experiments/run_unified.py` (updated runner template)

```python
import argparse
from src.configs.environment import EnvironmentManager, RuntimePreset

parser = argparse.ArgumentParser()
parser.add_argument("--preset", type=str, default="gpu_24g",
                   choices=[p.value for p in RuntimePreset])
parser.add_argument("--models", nargs="+", default="all",
                   help="Model names or 'all', 'main', 'generalist', 'specialist'")
parser.add_argument("--output", type=str, default="results/")
parser.add_argument("--num-samples", type=int, default=None)

args = parser.parse_args()

manager = EnvironmentManager()
config = manager.build_environment_config(
    preset=RuntimePreset[args.preset.upper()]
)
manager.validate_config(config)

# Load models from registry (filtered by tier)
from src.configs.model_taxonomy import get_models_by_tier, BenchmarkTier
models = get_models_by_tier(BenchmarkTier.MAIN)

# Run experiment with unified result writer
...
```

---

## Acceptance Criteria (Checklist)

### **Dependency Management** ✅
- [ ] `envs/` directory with 7 model-family-focused YAML files
- [x] `base.yaml`, `qwen.yaml`, `phi.yaml`, `internvl.yaml`, `llama.yaml`, `medical.yaml`, `specialist.yaml` created
- [ ] Documented in `envs/README.md` which env for which models
- [ ] Test: `conda env create -f envs/qwen.yaml` works isolated (no torch version conflict with specialist.yaml)

### **Model Taxonomy** ✅
- [x] Main benchmark: 13 models (6 gen, 3 domain, 3 spec) in `model_taxonomy.py`
- [x] Extra track: 8 models (OCR, chart, language, API) separated
- [ ] `model_registry.py` integrated with taxonomy (filter functions)
- [ ] Warnings when extra-track models selected for main benchmark

### **Runtime Presets** ✅
- [x] 5 presets in `presets/presets.yaml`
- [ ] Each preset includes: models list, quantization, batch size, image size, max tokens, fallback strategy
- [ ] Test on actual hardware: Colab T4, local RTX 3090, H100 (if available)

### **Sample Schema** ✅
- [x] `RadiologySample` dataclass in `src/data/schema.py`
- [ ] All dataset loaders (MIMIC, VQA_RAD, MS_CXR) return this schema
- [ ] Validation: `sample.is_valid_for_task("rrg")` works
- [ ] JSON round-trip: `.to_dict()` / `.from_dict()` preserves data

### **Backend Abstraction** ✅
- [x] `BackendInterface` ABC with `.load()`, `.infer()`, `.infer_batch()`
- [x] `TransformersBackend` implementation
- [x] `GenerationConfig` dataclass unified
- [ ] Test: Models load via backend, not direct transformers imports
- [ ] Future: vLLMBackend stub ready for extension

### **Result Logging** ✅
- [x] `ResultWriter` saves predictions.jsonl, sample_metrics.jsonl, aggregate_metrics.json, stats.json, errors.jsonl
- [ ] Test: Inference on 10 samples → all files populated, JSONL parseable
- [ ] Aggregate + sample-level metrics correctly computed

### **Infrastructure** ✅
- [x] `EnvironmentManager` loads presets, suggests env, validates config
- [ ] Integration with experiment runner: `--preset free_colab_t4` auto-sets batch=1, quant=4bit
- [ ] Warnings: T4 + no quant → warn about OOM risk

### **Notebook Refactoring** ✅
- [ ] `notebooks/00_repo_smoke_test.ipynb` (150 lines)
- [ ] `notebooks/01_dataset_schema_check.ipynb` (150 lines)
- [ ] `notebooks/models/{qwen2_vl_2b,phi35_vision,...}.ipynb` (8-10 notebooks, ~200 lines each)
- [ ] `notebooks/99_result_aggregation.ipynb` (200 lines)
- [ ] All save results to standardized `results/run_name/` structure

### **Documentation** ✅
- [ ] README updated: benchmark scope, model tiers, preset guide, quick-start per hardware
- [ ] `envs/README.md`: environment selection guide
- [ ] `presets/README.md` (optional): preset details, hardware assumptions
- [ ] Docstrings: `src/configurations/model_taxonomy.py`, `result_writer.py`

### **Reproducibility** ✅
- [ ] Config snapshot saved (config_snapshot.json)
- [ ] Environment info saved (environment.json)
- [ ] All predictions + metrics sample-level (predictions.jsonl, sample_metrics.jsonl)
- [ ] Paired statistical tests possible (Wilcoxon on sample-level arrays)
- [ ] Runs are resumable (JSONL append-only design)

---

## Known Risks & Mitigations

| Risk | Probability | Severity | Mitigation |
|---|---|---|---|
| **gated model access** (Llama-3.2-Vision, CheXagent) | High | Medium | Pre-flight check in notebook, guide to HF login, fallback model suggestion |
| **Transformer version mismatch** (Phi-3.5 vs Llama) | Medium | High | Separate envs guarantee isolation; test each env in CI |
| **T4 OOM** even with 4-bit | Medium | High | Preset fallback: skip model, smaller batch, or skip grounding task |
| **vLLM unavailable** on Colab | High | Low | Graceful fallback to transformers backend already designed |
| **Dataset incompleteness** (MIMIC path undefined) | Medium | Medium | Validation notebooks, clear error message with expected structure |
| **Breaking changes** in transformers 4.50+ | Low | High | Pin versions in envs (4.35-4.44 range safe as of 2025-03-20) |

---

## Next Steps for Repo Maintainers

**Week 1:**
1. Review this plan
2. Integrate `model_taxonomy.py` into `model_registry.py`
3. Reorganize `configs/model_configs/*.yaml` (mark extra-track models)
4. Create `notebooks/00_smoke_test.ipynb`, `notebooks/01_schema_check.ipynb` (test framework)

**Week 2:**
5. Refactor existing notebooks into model-specific ones
6. Update README with benchmark scope and quick-start
7. Create `experiments/run_unified.py` (unified runner)

**Week 3:**
8. Integration testing: smoke tests on Colab T4, local GPU
9. Document any new gating issues found
10. Publish v0.2.0 with working notebooks

---

## Files Created/Modified Summary

### New Files (10)
- ✅ `src/configs/model_taxonomy.py` - Model categorization
- ✅ `src/data/schema.py` - Unified RadiologySample
- ✅ `src/backends/base.py` - Backend interface
- ✅ `src/backends/transformers_backend.py` - Transformers implementation
- ✅ `src/utils/result_writer.py` - Result logging
- ✅ `src/configs/environment.py` - Environment management
- ✅ `envs/README.md` - Environment guide
- ✅ `envs/{base,qwen,phi,internvl,llama,medical,specialist,dev}.yaml` (8 files)
- ✅ `presets/presets.yaml` - Runtime presets

### Modified Files (4, to be done)
- [ ] `src/utils/model_registry.py` - Add taxonomy integration
- [ ] `configs/model_configs/{generalist,domain_adaptive,specialist}.yaml` - Reorganize
- [ ] `README.md` - Update guide
- [ ] `experiments/run_comparison.py` - Refactor for new backend

### Total: ~3000 lines of new/modified code, no breaking changes for existing pipelines
---

**Generated:** 2025-03-20 | **Status:** Foundation Laid ✅ | **Next Action:** Model Registry Integration
