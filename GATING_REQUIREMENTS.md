# Gating & Access Requirements - LLM Comparison

## Overview

Some models in our benchmark require special access setup or API keys. This document guides you through the requirements for each restricted model.

---

## 🔐 GATED MODELS (HuggingFace Access Required)

### Llama-3.2-Vision-11B

**Status:** Gated access on HuggingFace  
**Reason:** Meta's policy for Llama models  
**Fallback:** LLaVA-Med or Phi-3.5-Vision (MAIN benchmark)

#### Setup Steps:

1. **Create HuggingFace Account** (if needed)
   - Go to https://huggingface.co/join
   - Complete email and profile setup

2. **Accept Model License**
   - Navigate to: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
   - Click "Agree and access repository"
   - You'll need to accept the Meta Community License

3. **Get Hugging Face API Token**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Select "Read" access level (minimum)
   - Copy the token

4. **Login Locally**
   ```bash
   huggingface-cli login
   # When prompted, paste your token and press Enter
   ```
   - Token is saved to `~/.huggingface/hub/token`

5. **Verify Access**
   ```python
   from huggingface_hub import model_info
   try:
       info = model_info("meta-llama/Llama-3.2-11B-Vision-Instruct")
       print(f"✓ Access granted: {info.modelId}")
   except Exception as e:
       print(f"✗ Access denied: {e}")
   ```

---

### CheXagent-8B

**Status:** Gated access on HuggingFace  
**Reason:** Stanford research policy  
**Fallback:** LLaVA-Rad or RadFM (MAIN benchmark specialist models)

#### Setup Steps:

1. **Navigate to Model Page**
   - Go to: https://huggingface.co/StanfordAIMI/CheXagent-8b
   - Click "Agree and access repository"

2. **Proceed with Steps 3-5 from Llama-3.2-Vision above**

---

## 🔌 API MODELS (Cloud Service Access)

### GPT-4 Vision (OpenAI)

**Cost:** ~$0.01-0.03 per image inference  
**Status:** Requires paid OpenAI API key  
**Recommended for:** Optional comparison, additional validation

#### Setup Steps:

1. **Create OpenAI Account**
   - Go to https://platform.openai.com/signup
   - Set up billing method

2. **Generate API Key**
   - Navigate to: https://platform.openai.com/api-keys
   - Click "+ Create new secret key"
   - Copy the key immediately (only shown once)

3. **Set Environment Variable**
   ```bash
   # Linux/Mac
   export OPENAI_API_KEY="sk-..."
   
   # Windows PowerShell
   $env:OPENAI_API_KEY = "sk-..."
   
   # Add to .bashrc or .env for persistence
   echo "export OPENAI_API_KEY='sk-...'" >> ~/.bashrc
   ```

4. **Verify Access**
   ```python
   import os
   from openai import OpenAI
   
   try:
       client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
       print("✓ OpenAI API access verified")
   except Exception as e:
       print(f"✗ Access error: {e}")
   ```

5. **Monitor Costs**
   - Check usage at: https://platform.openai.com/account/usage/overview
   - Set spending limits to avoid surprises

---

### Gemini-1.5-Pro (Google)

**Cost:** ~$0.005-0.02 per image inference  
**Status:** Requires paid Google Cloud API key  
**Recommended for:** Optional comparison, additional validation

#### Setup Steps:

1. **Create Google Cloud Account**
   - Go to https://console.cloud.google.com
   - Create a new project

2. **Enable Gemini API**
   - Enable the "Google AI API" in your project
   - Set up billing

3. **Generate API Key**
   - Go to: https://ai.google.dev/tutorials/setup
   - Click "Get API key"
   - Select or create a Google Cloud project
   - Copy the API key

4. **Set Environment Variable**
   ```bash
   # Linux/Mac
   export GOOGLE_API_KEY="..."
   
   # Windows PowerShell
   $env:GOOGLE_API_KEY = "..."
   ```

5. **Verify Access**
   ```python
   import os
   import google.generativeai as genai
   
   try:
       genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
       print("✓ Google Gemini API access verified")
   except Exception as e:
       print(f"✗ Access error: {e}")
   ```

---

## ✓ Checking Model Access

Use the integrated access checker before experiments:

```python
from src.utils.model_registry import check_model_access

# Check a specific model
model_name = "llama3-vision"
access = check_model_access(model_name)

if access["accessible"]:
    print(f"✓ {model_name} is ready")
else:
    print(f"✗ {model_name}: {access['message']}")
    if "fix_url" in access:
        print(f"  Fix: {access['fix_url']}")
```

### Output Examples:

```
✓ qwen2-vl-2b is ready
✓ chexagent is ready

✗ gpt4v: Model 'gpt4v' requires OPENAI_API_KEY environment variable.
  Set: export OPENAI_API_KEY='your-key-here'
```

---

## 🚫 Common Issues & Solutions

### "Gated model - unauthorized"

**Problem:** Token is invalid or model access not granted

**Solutions:**
1. Verify token at: https://huggingface.co/settings/tokens
2. If token expired, create a new one
3. Make sure you accepted the model's license page
4. Try `huggingface-cli logout` then `huggingface-cli login` again

---

### "OPENAI_API_KEY not found"

**Problem:** Environment variable not properly set

**Solutions:**
```bash
# Check if set
echo $OPENAI_API_KEY

# Set (temporary, only for this shell session)
export OPENAI_API_KEY="sk-..."

# Set permanently (.bashrc or .zshrc)
echo "export OPENAI_API_KEY='sk-...'" >> ~/.bashrc
source ~/.bashrc

# Or create .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

---

### "Quota exceeded" (API models)

**Problem:** Hit usage limits

**Solutions:**
1. Check remaining quota at API dashboard
2. Set spending limits to prevent overage
3. Use `free_colab_t4` preset with open-source models instead
4. Reduce number of samples with `--num-samples 50`

---

## 📋 Quick Reference: Model Access Status

| Model | Type | Access | Config |  
|-------|------|--------|--------|
| **qwen2-vl-2b** | Local | ✓ Free | None |
| **qwen2.5-vl-3b** | Local | ✓ Free | None |
| **qwen3-vl-2b** | Local | ✓ Free | None |
| **phi3-vision** | Local | ✓ Free | None |
| **smolvlm2** | Local | ✓ Free | None |
| **internvl2-2b** | Local | ✓ Free | None |
| **internvl2-4b** | Local | ✓ Free | None |
| **llama3-vision** | Local | 🔐 Gated | HF Login |
| **llava-med** | Local | ✓ Free | None |
| **medgemma** | Local | ✓ Free | None |
| **biomedgpt** | Local | ✓ Free | None |
| **chexagent** | Local | 🔐 Gated | HF Login |
| **llava-rad** | Local | ✓ Free | None |
| **radfm** | Local | ✓ Free | None |
| **gpt4v** | API | 💳 Paid | OpenAI Key |
| **gemini** | API | 💳 Paid | Google Key |

---

## 📝 Notes

- **For MAIN benchmark (14 models):** Only 2 require special access (Llama-3.2, CheXagent)
- **For EXTRA track (7 models):** API models need keys; documented for reference
- **Budget consideration:** Running all 16 models with API keys could cost $50-100+ per experiment
- **Recommended:** Use FREE models for primary experiments, run API models on subset for validation

---

## Support & Troubleshooting

If you encounter access errors:

1. Run integrated checker: `python -c "from src.utils.model_registry import print_model_table; print_model_table()"`
2. Check detailed access status: `python examples/check_access.py`
3. Review environment setup: `bash setup.sh free_colab_t4` (will validate)

Good luck with your experiments! 🚀
