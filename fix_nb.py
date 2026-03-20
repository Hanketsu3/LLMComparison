import json

path = 'notebooks/run_full_experiment.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "# Test edilecek modeller (Colab T4 uyumlu, ücretsiz)\n",
    "MODELS_TO_TEST = [\n",
    "    # --- Generalist (Scaling Analysis) ---\n",
    "    \"qwen2-vl-2b\",       # 2B, Baseline\n",
    "    \"qwen2-vl-7b\",       # 7B, Scaling\n",
    "    # \"internvl2-2b\",      # 2B, (Şu an Transformers/Meta tensor hatası veriyor)\n",
    "    # \"internvl2-8b\",      # 8B, (Şu an Transformers/Meta tensor hatası veriyor)\n",
    "    # \"phi3-vision\",       # 4.2B, (DynamicCache hatası - yeni transformers uyumsuz)\n",
    "    # \"llama3-vision\",     # 11B, (HuggingFace Gated Repo - Erişim izni istiyor)\n",
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
        for i, line in enumerate(source):
            if line.startswith("MODELS_TO_TEST = ["):
                 # find the end of list
                 start_idx = i - 1 # include the comment above
                 end_idx = i
                 while not source[end_idx].startswith("]"):
                     end_idx += 1
                 
                 cell['source'] = source[:start_idx] + new_source + source[end_idx+2:]
                 break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
    f.write('\n')

print("Notebook updated successfully.")
