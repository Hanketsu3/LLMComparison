import json

with open('notebooks/run_full_experiment.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'MODELS_TO_TEST =' in ''.join(cell['source']):
        new_source = [
            '# ============================================================\n',
            '# DENEY KONFIGURASYONU\n',
            '# ============================================================\n',
            '\n',
            '# Test edilecek modeller (Colab T4 uyumlu, ucretsiz)\n',
            'MODELS_TO_TEST = [\n',
            '    # --- Generalist (Scaling Analysis) ---\n',
            '    # "qwen2-vl-2b",       # Tamamlandi (2B)\n',
            '    # "phi3-vision",       # Tamamlandi (4.2B)\n',
            '    # "internvl2-2b",      # Tamamlandi (2B)\n',
            '\n',
            '    "qwen2-vl-7b",       # 7B, 4-bit, scaling target\n',
            '    "internvl2-8b",      # 8B, 4-bit, scaling target\n',
            '    "llama3-vision",     # 11B, 4-bit, Meta\n',
            '    "llava-next-7b",     # 7B, 4-bit, LLaVA successor\n',
            '\n',
            '    # --- Domain-Adaptive ---\n',
            '    "llava-med",         # 7B, 4-bit, biomedical\n',
            '    "med-llama3-vision", # 11B, Llama3 Medical\n',
            '\n',
            '    # --- Specialist ---\n',
            '    "chexagent",         # 8B, 4-bit, Stanford AIMI\n',
            '    "llava-rad",         # 7B, 4-bit, radiology\n',
            ']\n',
            '\n',
            '# Deney parametreleri\n',
            'EXPERIMENT_NAME = "vqa_comparison_" + datetime.now().strftime("%Y%m%d_%H%M")\n',
            'RESULTS_DIR = Path("results") / EXPERIMENT_NAME\n',
            'RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n',
            '\n',
            'print(f"Deney: {EXPERIMENT_NAME}")\n',
            'print(f"Sonuclar: {RESULTS_DIR}")\n',
            'print(f"Test edilecek modeller ({len(MODELS_TO_TEST)}):")\n',
            'for m in MODELS_TO_TEST:\n',
            '    info = MODEL_REGISTRY.get(m)\n',
            '    if info:\n',
            '        print(f"   - {info.display_name} ({info.category.value}, {info.params})")\n',
            '\n',
            '# Kullanilacak model tablosu\n',
            'print("\\n" + "="*60)\n',
            'print_model_table()\n'
        ]
        cell['source'] = new_source
        break

with open('notebooks/run_full_experiment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('Notebook successfully updated.')
