#!/bin/bash
# Quick Start Setup Script for LLM Comparison Radiology Benchmark
# 
# This script sets up the environment for running experiments on different hardware.
# 
# USAGE:
#   bash setup.sh --preset free_colab_t4
#   bash setup.sh --preset gpu_24g
#   bash setup.sh --preset high_end_multi_gpu

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PRESET=${1:-gpu_24g}

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🏥 LLM Comparison - Radiology Benchmark Setup             ║"
echo "║     Preset: $PRESET                                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Validate preset
VALID_PRESETS=("smoke_cpu" "free_colab_t4" "colab_paid_mid" "gpu_24g" "high_end_multi_gpu")
if [[ ! " ${VALID_PRESETS[@]} " =~ " ${PRESET} " ]]; then
    echo -e "${RED}❌ Error: Invalid preset '$PRESET'${NC}"
    echo "   Valid presets: ${VALID_PRESETS[*]}"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

echo -e "${YELLOW}📦 Step 1: Detecting recommended environment...${NC}"

# Select environment based on preset
case $PRESET in
    smoke_cpu)
        ENV_FILE="envs/base.yaml"
        ENV_NAME="llmcomp-base"
        ;;
    free_colab_t4)
        ENV_FILE="envs/qwen.yaml"
        ENV_NAME="llmcomp-qwen"
        echo -e "${YELLOW}⚠️  Note: Colab T4 has ~12-14GB usable VRAM${NC}"
        echo -e "${YELLOW}   Models will use 4-bit quantization${NC}"
        ;;
    colab_paid_mid)
        ENV_FILE="envs/medical.yaml"
        ENV_NAME="llmcomp-medical"
        echo -e "${YELLOW}⚠️  Note: Colab paid assumes V100 (32GB) or A100-40GB${NC}"
        ;;
    gpu_24g)
        ENV_FILE="envs/specialist.yaml"
        ENV_NAME="llmcomp-specialist"
        echo -e "${YELLOW}✓ RTX 3090/4090 with radiologie specialist models${NC}"
        ;;
    high_end_multi_gpu)
        ENV_FILE="envs/specialist.yaml"
        ENV_NAME="llmcomp-specialist"
        echo -e "${YELLOW}✓ Multi-GPU A100/H100 cluster${NC}"
        ;;
esac

echo -e "${YELLOW}📦 Step 2: Checking conda...${NC}"

if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Conda not found. Please install Miniconda or Anaconda${NC}"
    echo "   https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html"
    exit 1
fi

echo -e "${GREEN}✓ Conda found$(conda --version)${NC}"

# Check if env already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}Environment '$ENV_NAME' already exists${NC}"
    read -p "Recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda remove -n $ENV_NAME --all -y
    else
        echo -e "${GREEN}✓ Using existing environment: $ENV_NAME${NC}"
        conda activate $ENV_NAME
        exit 0
    fi
fi

echo -e "${YELLOW}📦 Step 3: Creating conda environment from $ENV_FILE...${NC}"

if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ Environment file not found: $ENV_FILE${NC}"
    exit 1
fi

conda env create -f "$ENV_FILE" -n "$ENV_NAME" || {
    echo -e "${RED}❌ Failed to create environment${NC}"
    exit 1
}

echo -e "${YELLOW}📦 Step 4: Activating environment...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

echo -e "${YELLOW}📦 Step 5: Verifying installation...${NC}"

python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
python -c "from src.configs.environment import EnvironmentManager, RuntimePreset; print('✓ Custom modules importable')"

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ Setup Complete!                                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo "1️⃣  View preset details:"
echo "   python -c \"from src.configs.environment import EnvironmentManager, RuntimePreset; m = EnvironmentManager(); m.print_preset_summary(RuntimePreset.$PRESET)\""
echo ""
echo "2️⃣  Run smoke test:"
echo "   jupyter notebook notebooks/00_repo_smoke_test.ipynb"
echo ""
echo "3️⃣  Run a single model notebook (e.g., Qwen2-VL-2B):"
echo "   jupyter notebook notebooks/models/qwen2_vl_2b.ipynb"
echo ""
echo "4️⃣  Run full comparison (main benchmark models):"
echo "   python experiments/run_comparison.py --preset $PRESET --output results/"
echo ""
echo -e "${YELLOW}Environment activated: $ENV_NAME${NC}"
echo ""
