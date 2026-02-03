# 🏥 Radyoloji için Özgü vs Genel Amaçlı LLM Karşılaştırması

Bu proje, radyaloji alanında **Genel Amaçlı (Generalist)** ve **Alan Uyarlamalı/Uzman (Specialist)** Çok Modlu Büyük Dil Modellerinin (LMM) performanslarını karşılaştırmak için geliştirilmiş bir araştırma altyapısıdır.

Proje, hem akademik bir kıyaslama çalışması sunmakta hem de öğrencilerin/araştırmacıların kendi deneylerini yapabilmesi için hazır bir şablon sağlamaktadır.

---

## 🧪 Araştırma Kapsamı ve Hipotezler

Bu çalışma, aşağıdaki üç ana eksende modelleri kıyaslar:

### 1. Model Kategorileri
| Kategori | Tanım | Örnekler | Araştırma Sorusu |
|----------|-------|----------|------------------|
| **Generalist** | Tıp eğitimi almamış, devasa veriyle eğitilmiş genel modeller. | `Qwen2-VL`, `GPT-4V`, `Gemini` | Zero-shot yetenekleri uzman modelleri geçebilir mi? |
| **Domain-Adaptive** | Biyomedikal metinlerle (PubMed) eğitime devam edilmiş modeller. | `LLaVA-Med`, `Med-PaLM 2` | Terminoloji hakimiyeti görüntü yorumlamaya katkı sağlıyor mu? |
| **Specialist** | Doğrudan göğüs röntgeni ve rapor çiftleriyle eğitilmiş modeller. | `CheXagent`, `RadFM` | "Grounding" (yerelleştirme) konusunda en başarılılar mı? |

### 2. Değerlendirme Görevleri
Proje altyapısı aşağıdaki görevleri destekler:
*   **Rapor Üretimi (RRG):** Görüntüden tam teşekküllü radyoloji raporu yazma.
*   **Görsel Soru Cevaplama (VQA):** "Kalp boyutu normal mi?" gibi soruları yanıtlama.
*   **Yerelleştirme (Grounding):** Patolojilerin (örn. pnömoni) görüntü üzerindeki koordinatlarını bulma.

### 3. Başarı Metrikleri
*   **NLP:** BLEU, ROUGE (Metin benzerliği - *Not: Tıbbi doğruluk için yetersizdir, ancak baseline için kullanılır*).
*   **Klinik:** RadGraph F1 (Varlık ve ilişki doğruluğu - *Gold Standard*).
*   **Güvenilirlik:** Halüsinasyon Oranı ve Bias Testi.

---

## 📚 Teorik Arkaplan (Literatür Özeti)

Radyoloji raporu üretimi çalışmalarında 4 ana dönem bulunmaktadır:
1.  **Baseline Era (R2Gen):** CNN + Transformer kullanımı.
2.  **Knowledge-Driven:** Tıbbi bilgi grafikleri (RadGraph) ile destekleme.
3.  **RAG & Retrieval:** Benzer vakaları "kopya" çekerek halüsinasyonu azaltma.
4.  **SOTA (Multimodal LLM):** Chatbot şeklinde çalışan, yerelleştirme (grounding) yapabilen ajanlar.

**⚠️ Araştırma Tuzakları (Pitfalls):**
*   **Prior Bias:** Modelin görüntüye bakmadan "Akciğerler temiz" diye ezbere rapor yazması. *Çözüm: Boş görüntü testi.*
*   **Metrik Yanılgısı:** BLEU skorunun yüksek olması modelin klinik olarak doğru olduğunu göstermez.

---

## � Proje Yapısı

```
LLMComparison/
├── notebooks/             # Deney ortamı
│   └── main_experiment.ipynb  # <--- BAŞLANGIÇ NOKTASI (Öğrenci Şablonu)
├── src/                   # Kaynak kodlar
│   ├── models/            # Model entegrasyonları (Qwen2, CheXagent vb.)
│   ├── data/              # Veri yükleyiciler (MIMIC, VQA-RAD vb.)
│   ├── evaluation/        # Metrik hesaplamaları (RadGraph, GREEN)
│   └── utils/             # Yardımcı araçlar (Prompt yönetimi)
│       └── rag.py         # <--- YENİ: Retrieval-Augmented Generation iskeleti
├── configs/               # Deney konfigürasyonları (YAML)
│   └── experiment_configs/ # RRG, VQA, Grounding ayarları
├── experiments/           # Toplu deney scriptleri
└── results/               # Çıktıların kaydedildiği yer
```

---

## 🚀 Kurulum ve Kullanım (Öğrenci Kılavuzu)

Bu projeyi kullanmak için Google Colab önerilir.

### Adım 1: Projeyi İndirin
Projeyi Google Colab'da veya yerel ortamınızda klonlayın:
```bash
git clone https://github.com/Hanketsu3/LLMComparison.git
cd LLMComparison
```

### Adım 2: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 3: Deneye Başlayın
**`notebooks/main_experiment.ipynb`** dosyasını açın. Bu notebook, adım adım sizi yönlendirecektir:
1.  Kütüphaneleri yükleme ve kurulum.
2.  Açık kaynak bir model (Qwen2-VL) ile test yapma.
3.  Örnek bir röntgen görüntüsünü analiz etme.
4.  Farklı prompt tekniklerini (Baseline vs Detailed) karşılaştırma.

---

## 🔧 İleri Seviye Kullanım (Script ile)

Toplu deneyler için `experiments/run_comparison.py` scripti kullanılabilir. Hangi görevin çalışacağı `config` dosyasından belirlenir.

**Rapor Üretimi (RRG) Testi:**
```bash
python experiments/run_comparison.py --config configs/experiment_configs/rrg_experiment.yaml
```

**VQA Testi:**
```bash
python experiments/run_comparison.py --config configs/experiment_configs/vqa_experiment.yaml
```

---

## 🔮 Gelecek Çalışmalar (TODO)
- [ ] **MedGemma Entegrasyonu:** Google'ın açık kaynaklı MedGemma modeli ile kıyaslama ekle.
- [ ] **Daha Fazla Metrik:** BertScore ve BLEURT gibi semantik metrikleri dahil et.

---

## 👤 Katkıda Bulunanlar
*   **Proje Yürütücüsü:** Egemen Kaçıkan / Hanketsu3
*   **İletişim:** [Proje Linki](https://github.com/Hanketsu3/LLMComparison)
