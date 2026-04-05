# LLMComparison - Radiology VLM Benchmark

Bu repo, radyoloji odakli 3 gorev icin model karsilastirmasi yapar:
- RRG (Radiology Report Generation)
- VQA (Visual Question Answering)
- Grounding / localization

## Benchmark kapsami

Ana benchmark lane'leri:
- Generalist
- Domain-adaptive
- Specialist

Opsiyonel extra track:
- OCR / document parsing
- chart QA
- language adaptation
- API-only modeller

Model taxonomy tek noktadan yonetilir:
- src/utils/model_registry.py
- src/configs/model_taxonomy.py

## Model seti

Main (14):
- Generalist: qwen2-vl-2b, qwen2.5-vl-3b, qwen3-vl-2b, phi3-vision, smolvlm2-2.2b, internvl2-2b, internvl2-4b, llama3-vision
- Domain-adaptive: llava-med, medgemma-4b, biomedgpt
- Specialist: chexagent, llava-rad, radfm

Extra (7):
- got-ocr2, nougat-base, matcha-chartqa, qwen2-vl-ocr, latxa-qwen3-vl-2b, gpt4v, gemini

## Environment stratejisi

Model ailesine gore ayrik environment dosyalari:
- envs/generic_hf.yaml
- envs/qwen.yaml
- envs/phi.yaml
- envs/internvl.yaml
- envs/medical.yaml
- envs/specialist.yaml

Gated/API gereksinimleri icin:
- GATING_REQUIREMENTS.md

## Runtime presetleri

Presets:
- smoke_cpu
- free_colab_t4
- colab_paid_mid
- gpu_24g
- high_end_multi_gpu

Kaynak:
- presets/presets.yaml

Her preset su alanlari tanimlar:
- onerilen model listesi
- quantization
- batch size
- image_size / num_crops
- max_new_tokens
- attention/cache ayarlari
- riskler ve fallback stratejisi

## Unified pipeline

Yeni tek komutlu runner:
- experiments/run_unified.py

Akis:
- config -> dataset -> model -> inference -> sample-level metrics -> aggregate -> paired stats

Standart output dizini:
- results/<run_name>/config_snapshot.json
- results/<run_name>/predictions.jsonl
- results/<run_name>/sample_metrics.jsonl
- results/<run_name>/aggregate_metrics.json
- results/<run_name>/stats.json
- results/<run_name>/errors.jsonl
- results/<run_name>/environment.json

## Notebook mimarisi

Model notebooklari:
- notebooks/models/qwen2_vl_2b.ipynb
- notebooks/models/qwen25_vl_3b.ipynb
- notebooks/models/phi35_vision.ipynb
- notebooks/models/internvl2_2b.ipynb
- notebooks/models/medgemma_4b.ipynb
- notebooks/models/chexagent.ipynb
- notebooks/models/radfm.ipynb
- notebooks/models/llava_med.ipynb

Family notebooklari:
- notebooks/families/qwen_family.ipynb
- notebooks/families/phi_family.ipynb
- notebooks/families/internvl_family.ipynb

Repo-level notebooklar:
- notebooks/00_repo_smoke_test.ipynb
- notebooks/01_dataset_schema_check.ipynb
- notebooks/99_result_aggregation.ipynb
- notebooks/aggregation_analysis.ipynb

Karsilastirma notebooklari (yalnizca saved outputs):
- notebooks/main_experiment.ipynb
- notebooks/paper_experiment.ipynb
- notebooks/run_full_experiment.ipynb

## Calistirma komutlari

Smoke test:
```bash
python experiments/run_unified.py \
  --preset smoke_cpu \
  --models qwen2-vl-2b \
  --datasets hf_vqa_rad \
  --num-samples 5 \
  --skip-inaccessible \
  --output-dir results \
  --run-name smoke_cpu_qwen
```

Colab T4:
```bash
python experiments/run_unified.py \
  --preset free_colab_t4 \
  --models generalist \
  --datasets hf_vqa_rad \
  --num-samples 50 \
  --skip-inaccessible \
  --output-dir results \
  --run-name colab_t4_generalist
```

24GB GPU:
```bash
python experiments/run_unified.py \
  --preset gpu_24g \
  --models main \
  --datasets hf_iu_xray hf_vqa_rad \
  --num-samples 120 \
  --skip-inaccessible \
  --output-dir results \
  --run-name gpu24_main
```

## Common failure cases

- access_gated
- missing_api_key
- insufficient_vram
- missing_vllm
- incompatible transformers version

## Reproducibility

- Her kosunun config ve environment snapshot'i kaydedilir.
- Sample-level prediction ve metric kaydi tutulur.
- Paired statistics sample-level dizi uzerinden hesaplanir.

## Testler

Fast test seti:
```bash
pytest tests/test_registry.py tests/test_runtime_config.py tests/test_schema_normalization.py tests/test_unified_runner_smoke.py tests/test_result_writer.py -q
```
