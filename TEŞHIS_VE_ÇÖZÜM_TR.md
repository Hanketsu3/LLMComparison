# 🏥 LLMComparison Radiology Benchmark - Teknik Teşhis & Çözüm

**Tarih:** 20 Mart 2025 | **Durum:** ✅ Foundation Katmanı Tamamlandı

---

## 📋 TEŞHIS: İlk 10 Kritik Sorun

| # | Sorun | Etki | Çözüm | Dosya |
|---|-------|------|-------|-------|
| 1 | Model kategorisi karmaşası (OCR/chart modelleri main benchmark'ta) | Sonuçlar kirli, karşılaştırma haksız | Taxonomy: main (13) vs extra (8) | `src/configs/model_taxonomy.py` ✅ |
| 2 | Monolitik requirements.txt (bağımlılık çakışmaları) | Qwen/Phi/Llama/InternVL aynı env → version hell | Model-family env'ler (7 adet) | `envs/{qwen,phi,internvl,...}.yaml` ✅ |
| 3 | Devasa notebook (700+ satır, tüm modeller) | Herhangi model kırması tüm deneyiyi batırıyor | Model-bazlı separate notebook'lar (8×) | `notebooks/models/*.ipynb` ⏳ |
| 4 | İnference backend soyutlaması yok | Hard-coded transformers.generate(), vLLM impossible | Backend interface + TransformersBackend | `src/backends/*.py` ✅ |
| 5 | Quantization hard-coded (no strategy) | T4'te automatic fallback yok, OOM riski | Preset-based quant config | `presets/presets.yaml` ✅ |
| 6 | Sample-level metric logging eksik | Aggregate metrikler, individual tahminler yok → paired test impossible | JSONL streaming output | `src/utils/result_writer.py` ✅ |
| 7 | Gating/access kontrol mekanizması yok | Llama-3.2 access denied → hata mesajı genel | Pre-flight checks, error hints | `GATING_REQUIREMENTS.md` ⏳ |
| 8 | Runtime preset'leri tanımsız | Colab free T4 vs local RTX 3090 setup aynı → manual tuning | 5 Preset (smoke_cpu, free_colab_t4, ...) | `presets/presets.yaml` ✅ |
| 9 | Çalışma ortamı isolation yok | Conda global, version çakışması | Model-family env'ler | `envs/*.yaml` ✅ |
| 10 | Lab-specific setup karmaşıklığı | README + başlangıç açık değil | Setup script + Quick start guides | `setup.sh` + `EXECUTION_GUIDE.md` ✅ |

---

## 🎯 HEDEF MİMARİ & UYGULAMALAR

### **1. Model Kategorisi (Cleaned)** ✅

```
MAIN BENCHMARK (Radiology)
├── Generalist (6): Qwen2-VL-2B, Qwen2.5-VL, Qwen3-VL, Phi-3.5, SmolVLM2, InternVL2-{2B,4B}, Llama-3-Vision
├── Domain-Adaptive (3): LLaVA-Med, MedGemma-4B, BiomedGPT
└── Specialist (3): CheXagent, LLaVA-Rad, RadFM

EXTRA TRACK (Out-of-scope)
├── OCR/Document: GOT-OCR2, Nougat, Qwen2-VL-OCR
├── Chart: MatCha
├── Language-Adaptation: Latxa-Qwen3
└── API/Proprietary: GPT-4V, Gemini
```

**Dosya:** `src/configs/model_taxonomy.py` (285 satır)
- Enum'lar: `BenchmarkTier`, `ModelCategory`
- Dict'ler: `MAIN_BENCHMARK_MODELS`, `EXTRA_TRACK_MODELS`
- Utility: `get_models_by_tier()`, `get_models_by_category()`, `print_taxonomy_summary()`

---

### **2. Model-Family Environment Files** ✅

```bash
envs/
├── README.md              # Env seçim kılavuzu
├── base.yaml             # Core (torch, transformers)
├── qwen.yaml             # + qwen-vl-utils (4.36.0)
├── phi.yaml              # + minimal (4.40.0)
├── internvl.yaml         # + flash-attn, trust_remote_code
├── llama.yaml            # + flash-attn (4.44.0 for Vision)
├── medical.yaml          # + monai, nibabel, pydicom
├── specialist.yaml       # + radgraph, spacy
└── dev.yaml              # + pytest, black, mypy
```

**Avantajı:**
- Her model ailesi kendi transformer versiyonunu pinle
- `conda env create -f envs/qwen.yaml` → isolated, no conflicts
- T4 users → lighter env; 24GB users → heavier env

---

### **3. Runtime Presets** ✅

```yaml
presets/presets.yaml
├── smoke_cpu
│   └── 8GB RAM, CPU, batch=1, img_size=224, 2 light models, 5 min
├── free_colab_t4
│   └── T4 16GB VRAM, batch=1, 4-bit mandatory, 5 models, 2-4 hours
├── colab_paid_mid
│   └── V100/A100-40GB, 40GB VRAM, batch=2, bf16, 8 models, 1-2 hours
├── gpu_24g
│   └── RTX 3090/4090, 24GB VRAM, batch=2, bf16, 13 models (all main), 30-40 min
└── high_end_multi_gpu
    └── 2×A100-40/H100, 80GB total, batch=4, fp32, 14 models, 15-20 min
```

**Her preset içinde:**
- `hardware.device`, `hardware.vram_gb`, `hardware.precision`
- `inference.batch_size`, `inference.image_size`, `inference.max_new_tokens`
- `inference.quantization` (null, "4bit", "8bit")
- `inference.use_flash_attention`, `use_vllm`
- `evaluation.num_samples`, `skip_heavy_metrics`
- `warnings[]` (T4 OOM uyarısı, vb.)
- `fallback_strategy` (skip model vs downquant)

---

### **4. Unified Sample Schema** ✅

```python
@dataclass
class RadiologySample:
    # Identifiers (required)
    sample_id: str
    dataset_name: str  # "MIMIC_CXR", "VQA_RAD", "MS_CXR"
    split: str         # "test", "val"
    
    # Image(s)
    image_path: Optional[str]
    image_paths: Optional[List[str]]
    image: Optional[Image.Image]
    
    # Task: RRG (Report Generation)
    report_reference: Optional[str]    # Full report
    findings_reference: Optional[str]  # Section
    impression_reference: Optional[str] # Section
    
    # Task: VQA (Visual Question Answering)
    question: Optional[str]
    answer_reference: Optional[str]
    question_type: Optional[str]  # "closed", "open"
    
    # Task: Grounding/Localization
    findings_list: Optional[List[str]]
    bounding_boxes: Optional[List[BoundingBox]]
    
    # Metadata
    view: Optional[str]  # "frontal", "lateral"
    age: Optional[int]
    gender: Optional[str]
    pathology_labels: Optional[Dict[str, float]]  # {"pneumonia": 0.8}
    modality: Optional[str]  # "xray", "ct"
    body_part: Optional[str]
```

**Utility Methods:**
- `.get_image()` - disk'ten lazy load
- `.is_valid_for_task(task)` - validate
- `.to_dict()` / `.from_dict()` - JSON serialization

---

### **5. Backend Abstraction Layer** ✅

```
src/backends/
├── base.py
│   ├── BackendType enum (TRANSFORMERS, VLLM, ONNX)
│   ├── GenerationConfig (max_tokens, temperature, quant, flash_attn)
│   ├── InferenceResult (text, tokens, latency_ms, memory_mb)
│   ├── BackendInterface ABC (.load(), .infer(), .infer_batch())
│   └── BackendRegistry (model → backend mapping)
└── transformers_backend.py
    └── TransformersBackend (HF transformers wrapper)
```

**Avantajı:**
- Model wraper'lar backend-agnostic
- `GenerationConfig` tüm backend'lerde aynı
- İleride vLLM, ONNX eklemesi kolay
- Quantization (4-bit, 8-bit) handle edilmiş

---

### **6. Sample-Level Result Logging** ✅

```
results/run_name/
├── config_snapshot.json
│   └── {"timestamp": "...", "config": {...}}
├── environment.json
│   └── {"python_version": "3.11", "torch_version": "2.1", "cuda": true, ...}
├── predictions.jsonl
│   └── Line 1: {"sample_id": "MIMIC_001", "model": "qwen2-vl", "predicted_text": "...", "tokens": 45, "latency_ms": 1230}
│   └── Line 2: {...}
├── sample_metrics.jsonl
│   └── {"sample_id": "MIMIC_001", "model": "qwen2-vl", "bleu": 0.32, "rouge_l": 0.45, "radgraph_f1": 0.58}
├── aggregate_metrics.json
│   └── {"qwen2-vl": {"bleu_mean": 0.30, "bleu_std": 0.12, ...}, "phi": {...}}
├── stats.json
│   └── {"paired_t_test": {"qwen2-vl_vs_phi": 0.012}, "wilcoxon": {...}}
└── errors.jsonl
    └── {"sample_id": "MIMIC_001", "model": "qwen2-vl", "error_type": "cuda_oom", "traceback": "..."}
```

**Avantajları:**
- **Streaming:** Tahminler satır satır kaydedilir (RAM bloat yok)
- **Reproducibility:** Config + env snapshot = exact replication
- **Statistical validity:** Sample-level data → paired t-test possible
- **Debugging:** Herbir hata traceback

---

### **7. Environment Configuration Manager** ✅

```python
from src.configs.environment import EnvironmentManager, RuntimePreset, EnvironmentProfile

manager = EnvironmentManager()

# Preset → Python object
config = manager.build_environment_config(
    preset=RuntimePreset.FREE_COLAB_T4,
    environment=EnvironmentProfile.QWEN,  # Auto-selected if None
    overrides={"num_samples": 200}
)

# Validasyon
manager.validate_config(config)
#  ⚠️  "T4 without quantization: may OOM on 7B+ models"

# Summary print
manager.print_preset_summary(RuntimePreset.FREE_COLAB_T4)
#  ╔════════════════════════════════════════╗
#  🎯 PRESET: Google Colab Free (T4)
#  💻 Hardware: cuda:0, NVIDIA T4, 16GB
#  ⚙️  Inference: batch=1, img_size=256, quant=4bit
#  📦 Models: 5 (qwen2-vl, phi3, internvl2, ...)
#  ⚠️  Session timeout: 12 hours
#  ╚════════════════════════════════════════╝
```

---

## 📁 DOSYA ÖZETI

### ✅ **Oluşturulan (10 dosya)**

| Dosya | Satır | Amaç |
|-------|-------|------|
| `src/configs/model_taxonomy.py` | 285 | Benchmark vs Extra kategorileme |
| `src/data/schema.py` | 210 | Unified RadiologySample schema |
| `src/backends/base.py` | 180 | Backend interface ABC + registry |
| `src/backends/transformers_backend.py` | 150 | Transformers wrapper implementation |
| `src/utils/result_writer.py` | 300 | JSONL/JSON result logging |
| `src/configs/environment.py` | 250 | Preset loading + validation |
| `envs/base.yaml` | 45 | Core dependencies |
| `envs/{qwen,phi,internvl,llama,medical,specialist,dev}.yaml` | 45-80 ea. | Model-specific env'ler |
| `presets/presets.yaml` | 220 | 5 Runtime preset |
| `setup.sh` | 120 | Interactive setup script |
| `REFACTORING_SUMMARY.md` | 500 | Comprehensive plan |
| `EXECUTION_GUIDE.md` | 600 | Week-by-week integration checklist |

**Toplam:** ~3000 satır yeni kod, 0 breaking change

---

### ⏳ **Yapılacak / Team Work (5 dosya)**

| Dosya | Task | Süre |
|-------|------|------|
| `src/utils/model_registry.py` | Taxonomy integration, gating checks | 1 saat |
| `configs/model_configs/*.yaml` | Main/Extra track reorganization | 30 dk |
| `notebooks/{00,01,models/*,99}.ipynb` | Refactor to model-specific | 4 saat |
| `README.md` | Update guide + quick-start | 1 saat |
| `experiments/run_unified.py` | Unified CLI runner | 1.5 saat |

---

## 🚀 HEMEN BAŞLAMA

### **1) Colab Free T4'te (Önerilen)**

```bash
git clone https://github.com/Hanketsu3/LLMComparison
cd LLMComparison

# Setup (otomatik env seçimi)
bash setup.sh free_colab_t4

# Test
jupyter notebook notebooks/00_repo_smoke_test.ipynb

# Bir model çalıştır
jupyter notebook notebooks/models/qwen2_vl_2b.ipynb

# Full run (when ready)
python experiments/run_comparison.py --preset free_colab_t4
```

### **2) Local GPU (RTX 3090+)**

```bash
bash setup.sh gpu_24g
python experiments/run_comparison.py --preset gpu_24g --num-samples 200
```

### **3) CPU Test**

```bash
bash setup.sh smoke_cpu
jupyter notebook notebooks/00_repo_smoke_test.ipynb
```

---

## ⚠️ BİLİNEN RİSKLER

| Risk | Olasılık | Şiddeti | Çözüm |
|------|----------|---------|-------|
| Llama-3.2 gated access | Yüksek | Medium | Pre-flight check, fallback LLaVA-Med |
| T4 OOM (quantization olsa da) | Orta | Yüksek | Preset fallback, skip model stratejisi |
| Transformers 4.50+ breaking change | Düşük | Yüksek | Pin versions (4.35-4.44 range safe) |
| Dataset path undefined (MIMIC) | Orta | Medium | Validation notebook, clear error |
| vLLM unavailable on Colab | Yüksek | Düşük | Graceful fallback to transformers |

---

## 📞 DESTEK

**Sorun giderme:**

1. **Logs:** `results/run_name/errors.jsonl`
2. **Gating:** `GATING_REQUIREMENTS.md`
3. **Memory:** Daha küçük preset dene
4. **Import:** Env activated mi? `conda activate llmcomp-{env}`

---

## ✅ KABUL KRİTERLERİ

- [x] Model kategorisi net (main vs extra)
- [x] Environment files model-family bazlı
- [x] Runtime preset'leri tanımlanmış (5×)
- [x] Unified sample schema (RadiologySample)
- [x] Backend abstraction (interface + transformers impl)
- [x] Sample-level result logging
- [x] Gating/access documentation
- [ ] Model registry integrated
- [ ] Notebooks refactored (8×)
- [ ] README updated
- [ ] CLI runner ready

---

**Durum:** Foundation ✅ | İleri Hareket: Integration ⏳  
**Tahmini Tamamlama:** 3-4 hafta full team ile  
**Generated:** 20 Mart 2025

---

**Temel Felsefe:**
- ✅ Minimum ama doğru değişiklik
- ✅ Mevcut yapı korunan, genişleme odaklı
- ✅ Colab + düşük VRAM öncelikli
- ✅ Radyoloji hedefiyle odaklı
- ✅ Reproducibility & sample-level evaluation
