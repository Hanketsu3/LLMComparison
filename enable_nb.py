import json

path = 'notebooks/run_full_experiment.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "# Test edilecek modeller (Colab T4 uyumlu, ücretsiz)\n",
    "MODELS_TO_TEST = [\n",
    "    # --- Generalist (Scaling Analysis) ---\n",
    "    \"qwen2-vl-2b\",       # 2B, Baseline (Çalışıyor: %43)\n",
    "    \"qwen2-vl-7b\",       # 7B, Scaling (Çalışıyor: %55)\n",
    "    \"internvl2-2b\",      # 2B, (Meta tensor hatası düzeltildi!)\n",
    "    \"internvl2-8b\",      # 8B, (Meta tensor hatası düzeltildi!)\n",
    "    \"phi3-vision\",       # 4.2B, (DynamicCache hatası düzeltildi!)\n",
    "    \"llama3-vision\",     # 11B, (DİKKAT: HuggingFace izni gerekir!)\n",
    "    \"llava-next-7b\",     # 7B, LLaVA successor\n",
    "\n",
    "    # --- Domain-Adaptive ---\n",
    "    \"llava-med\",         # 7B, Biomedical\n",
    "    \"med-llama3-vision\", # 11B, Llama3 Medical\n",
    "\n",
    "    # --- Specialist ---\n",
    "    \"chexagent\",         # 8B, Stanford AIMI\n",
    "    \"llava-rad\",         # 7B, Radiology\n",
    "]\n"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if len(source) > 0 and source[0].startswith("# Test edilecek modeller"):
            # find the end of list
            start_idx = 0
            end_idx = 0
            for i, line in enumerate(source):
                if "]" in line and "MODELS_TO_TEST" not in line: # simplistic check
                    end_idx = i
                    break
            
            cell['source'] = new_source + source[end_idx+1:]
            break

# Add HF Login instruction cell right before it
login_cell_md = {
  "cell_type": "markdown",
  "metadata": {},
  "source": [
    "---\n",
    "## 🔑 ÖNEMLİ: HuggingFace Gated Repolar İçin Giriş\n",
    "`llama3-vision` gibi kapalı (gated) repoları indirmek için HF token'ına ihtiyacın var. Eğer bu modeli çalıştıracaksan aşağıdaki hücreyi UNCOMMENT yap ve token'ını gir:"
  ]
}

login_cell_code = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "outputs": [],
  "source": [
    "# !pip install -U huggingface_hub\n",
    "# from huggingface_hub import notebook_login\n",
    "# notebook_login()"
  ]
}

# Insert after cell 6 (or somewhere before experiment config)
nb['cells'].insert(8, login_cell_md)
nb['cells'].insert(9, login_cell_code)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
    f.write('\n')

print("Notebook un-disabled successfully.")
