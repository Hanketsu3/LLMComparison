# Environment File Documentation
# ==============================================================================
# This directory contains conda/pip environment specifications for different
# model families to avoid dependency conflicts.
#
# USAGE:
# ------
# For conda:
#   conda env create -f base.yaml -n llmcomp-base
#   conda env create -f qwen.yaml -n llmcomp-qwen
#
# For pip (with venv):
#   python -m venv .venv
#   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
#   pip install -r <(echo "# placeholder") $(cat base.yaml | grep "^  -" | cut -d'-' -f2)
#
# WHICH ENV TO CHOOSE:
# --------------------
# Model                    Environment        Notes
# ─────────────────────────────────────────────────────────────────────────
# Qwen2/3-VL series        qwen.yaml          Requires qwen-vl-utils
# Phi-3.5-Vision           phi.yaml           Minimal special deps
# InternVL2 series         internvl.yaml      Requires trust_remote_code
# Llama-3.2-Vision         llama.yaml         Gated model, needs token
# LLaVA-Med                medical.yaml       Medical libs
# CheXagent / LLaVA-Rad    specialist.yaml    Radiology-specific
# MedGemma, BiomedGPT      medical.yaml       Medical domain
# All else / development   base.yaml          Core transformers, torch
#
# TRANSFORMER VERSION PINNING:
# ----------------------------
# - base.yaml:       transformers>=4.35.0
# - qwen.yaml:       transformers>=4.36.0 (Qwen2-VL processor fix)
# - phi.yaml:        transformers>=4.40.0 (Phi-3.5 recent fix)
# - internvl.yaml:   transformers>=4.35.0 + torch>=2.0.0
# - llama.yaml:      transformers>=4.44.0 (Llama-3.2-Vision)
# - medical.yaml:    transformers>=4.35.0 + medical libs
# - specialist.yaml: transformers>=4.35.0 + radiology libs
#
# PYTORCH/CUDA VERSIONS:
# ----------------------
# If using Ampere (RTX 30-series, A5000+): torch~=2.1.0, cuda==11.8
# If using Hopper (H100, L40S):             torch~=2.2.0, cuda==12.1
# If using T4 (Colab):                      torch==2.0.1, cuda==11.8
# If using CPU:                             torch CPU build only
#
# TORCH INSTALLATION (example for CUDA 11.8):
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#
# GPP VERSION CONFLICTS:
# ----------------------
# Qwen2-VL + Llama3.2-Vision:  Avoid same env (transformers conflict)
# Phi-3.5 + InternVL2:         Can coexist in medical.yaml
# CheXagent + LLaVA-Rad:       Both 8B, use specialist.yaml
#
# UPDATING ENVS:
# ---------------
# If a model fails:
#   1. Check error type (import, version, cuda, memory)
#   2. Update conda-forge / pip index in the relevant env file
#   3. Test isolated model in dedicated notebook
#
# ============================================================================
