#!/bin/bash
# Setup models for the LLM Comparison project

set -e

echo "=== LLM Comparison Model Setup Script ==="
echo ""

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN not set. Some models may require authentication."
    echo "   Set HF_TOKEN environment variable for gated models."
fi

# Login to HuggingFace
if command -v huggingface-cli &> /dev/null; then
    if [ -n "$HF_TOKEN" ]; then
        echo "🔐 Logging in to HuggingFace..."
        huggingface-cli login --token "$HF_TOKEN"
    fi
fi

# Download CheXagent
download_chexagent() {
    echo "📦 Downloading CheXagent..."
    
    python -c "
from transformers import AutoModelForCausalLM, AutoProcessor

print('Downloading CheXagent-8b...')
processor = AutoProcessor.from_pretrained(
    'StanfordAIMI/CheXagent-8b',
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    'StanfordAIMI/CheXagent-8b',
    trust_remote_code=True,
    torch_dtype='auto'
)
print('✅ CheXagent downloaded successfully')
"
}

# Download LLaVA-Med
download_llava_med() {
    echo "📦 Downloading LLaVA-Med..."
    
    python -c "
from transformers import AutoProcessor, LlavaForConditionalGeneration

print('Downloading LLaVA-Med...')
processor = AutoProcessor.from_pretrained('microsoft/llava-med-v1.5-mistral-7b')
model = LlavaForConditionalGeneration.from_pretrained(
    'microsoft/llava-med-v1.5-mistral-7b',
    torch_dtype='auto'
)
print('✅ LLaVA-Med downloaded successfully')
"
}

# Main
echo "Which models would you like to download?"
echo "1) CheXagent (Specialist)"
echo "2) LLaVA-Med (Domain-Adaptive)"
echo "3) All"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1) download_chexagent ;;
    2) download_llava_med ;;
    3) 
        download_chexagent
        download_llava_med
        ;;
    *) echo "Invalid choice" ;;
esac

echo ""
echo "=== Model setup complete ==="
