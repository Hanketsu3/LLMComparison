# LLM Comparison: Domain-Specific vs General-Purpose Models for Radiology

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PhD seviyesinde araştırma projesi: Radyoloji alanında özgü (domain-specific) ve genel amaçlı (general-purpose) büyük dil modellerinin karşılaştırmalı analizi.

## 📋 Araştırma Özeti

Bu proje, üç ana model kategorisinin radyoloji görevlerindeki performansını karşılaştırmaktadır:

| Kategori | Modeller | Özellikler |
|----------|----------|------------|
| **Genel Amaçlı (Generalist)** | GPT-4V, Gemini 1.5 Pro, Llama-3 | Tıp eğitimi almamış, yüksek zero-shot yetenekler |
| **Alan Uyarlamalı (Domain-Adaptive)** | Med-PaLM 2, LLaVA-Med | Biyomedikal literatürle eğitilmiş |
| **Radyoloji Uzmanı (Specialist)** | CheXagent, MAIRA-2, LLaVA-Rad, RadFM | Görüntü-rapor çiftleriyle eğitilmiş |

## 🎯 Değerlendirme Görevleri

1. **Rapor Üretimi (RRG)**: MIMIC-CXR, IU X-Ray veri setleri
2. **Görsel Soru Cevaplama (VQA)**: VQA-RAD, SLAKE, PathVQA
3. **Yerelleştirme (Grounding)**: MS-CXR, VinDr-CXR

## 📊 Metrikler

### Klinik Metrikler (Gold Standard)
- **RadGraph F1**: Varlık ve ilişki doğruluğu
- **CheXbert F1**: Chest X-ray bulgu sınıflandırması

### LLM Tabanlı Hakem
- **GREEN**: Klinik hata değerlendirmesi
- **RadCliQ**: Radyoloji kalite skoru
- **RadFact**: Olgusal doğruluk

### Halüsinasyon Tespiti
- **FactCheXcker**: Uydurma bulgu tespiti
- **Object Hallucination**: Nesne halüsinasyonu oranı

## 🚀 Kurulum

### Gereksinimler
- Python 3.10+
- CUDA 11.8+ (GPU modelleri için)
- 16GB+ RAM (uzman modeller için 32GB+ önerilir)

### Kurulum Adımları

```bash
# Repository'yi klonlayın
git clone https://github.com/yourusername/LLMComparison.git
cd LLMComparison

# Virtual environment oluşturun
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya: venv\Scripts\activate  # Windows

# Bağımlılıkları yükleyin
pip install -e .

# API anahtarlarını yapılandırın
cp .env.example .env
# .env dosyasını düzenleyerek API anahtarlarınızı ekleyin
```

### Veri Seti Kurulumu

```bash
# Veri setlerini indirin (PhysioNet hesabı gerekli)
./scripts/download_datasets.sh

# Modelleri indirin
./scripts/setup_models.sh
```

## 📁 Proje Yapısı

```
LLMComparison/
├── configs/           # Yapılandırma dosyaları
├── src/               # Ana kaynak kodu
│   ├── data/          # Veri seti yönetimi
│   ├── models/        # Model sarmalayıcıları
│   ├── encoders/      # Görsel kodlayıcılar
│   ├── evaluation/    # Değerlendirme metrikleri
│   ├── tasks/         # Görev modülleri
│   └── utils/         # Yardımcı fonksiyonlar
├── experiments/       # Deney çalıştırıcıları
├── notebooks/         # Jupyter notebooklar
├── scripts/           # Yardımcı scriptler
├── tests/             # Birim testleri
└── results/           # Sonuçlar
```

## 🔬 Deney Çalıştırma

### 1. Baseline Testi (R2Gen)

```bash
python experiments/run_baseline.py \
    --config configs/experiment_configs/rrg_experiment.yaml \
    --dataset mimic-cxr \
    --split test
```

### 2. Genel Model Testi (GPT-4V)

```bash
python experiments/run_generalist.py \
    --model gpt4v \
    --config configs/model_configs/generalist.yaml \
    --few-shot 3
```

### 3. Uzman Model Testi (CheXagent)

```bash
python experiments/run_specialist.py \
    --model chexagent \
    --config configs/model_configs/specialist.yaml
```

### 4. Karşılaştırma Analizi

```bash
python experiments/run_comparison.py \
    --models baseline,gpt4v,chexagent \
    --metrics radgraph_f1,green,hallucination \
    --output results/comparison_results.json
```

## 📈 Sonuçların Analizi

Jupyter notebook'ları kullanarak sonuçları analiz edin:

```bash
jupyter notebook notebooks/03_model_comparison.ipynb
```

## 🔒 Veri Gizliliği

> ⚠️ **Önemli**: Hasta görüntülerini bulut API'larına (GPT-4, Gemini) göndermekten kaçının. Yerel modelleri (Llama-3 tabanlı) tercih edin.

## 📚 Referanslar

- [RadGraph](https://physionet.org/content/radgraph/1.0.0/) - Radyoloji varlık ve ilişki çıkarımı
- [CheXagent](https://stanford-aimi.github.io/chexagent.html) - Stanford AIMI radyoloji modeli
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) - Chest X-ray veri seti

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👤 İletişim

Sorularınız için: [email protected]
