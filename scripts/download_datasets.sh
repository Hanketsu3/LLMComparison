#!/bin/bash
# Download datasets for the LLM Comparison project

set -e

echo "=== LLM Comparison Dataset Download Script ==="
echo ""

# Create data directory
DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "$DATA_DIR"

echo "Data will be downloaded to: $DATA_DIR"
echo ""

# Function to check if PhysioNet credentials are set
check_physionet_credentials() {
    if [ -z "$PHYSIONET_USERNAME" ] || [ -z "$PHYSIONET_PASSWORD" ]; then
        echo "⚠️  PhysioNet credentials not set."
        echo "   Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD environment variables."
        echo "   You need a PhysioNet account with credentialed access to MIMIC-CXR."
        echo "   Register at: https://physionet.org/register/"
        return 1
    fi
    return 0
}

# MIMIC-CXR
download_mimic_cxr() {
    echo "📦 Downloading MIMIC-CXR..."
    
    if ! check_physionet_credentials; then
        echo "   Skipping MIMIC-CXR download."
        return
    fi
    
    MIMIC_DIR="$DATA_DIR/mimic-cxr"
    mkdir -p "$MIMIC_DIR"
    
    # Download using wget
    wget -r -N -c -np \
        --user "$PHYSIONET_USERNAME" \
        --password "$PHYSIONET_PASSWORD" \
        https://physionet.org/files/mimic-cxr/2.0.0/ \
        -P "$MIMIC_DIR"
    
    echo "✅ MIMIC-CXR downloaded to $MIMIC_DIR"
}

# VQA-RAD
download_vqa_rad() {
    echo "📦 Downloading VQA-RAD..."
    
    VQA_DIR="$DATA_DIR/vqa-rad"
    mkdir -p "$VQA_DIR"
    
    # VQA-RAD is available from OSF
    # https://osf.io/89kps/
    echo "   VQA-RAD must be downloaded manually from: https://osf.io/89kps/"
    echo "   Please download and extract to: $VQA_DIR"
}

# SLAKE
download_slake() {
    echo "📦 Downloading SLAKE..."
    
    SLAKE_DIR="$DATA_DIR/slake"
    mkdir -p "$SLAKE_DIR"
    
    # SLAKE from GitHub
    if [ ! -d "$SLAKE_DIR/.git" ]; then
        git clone https://github.com/Slake-datasets/Slake.git "$SLAKE_DIR"
    else
        cd "$SLAKE_DIR" && git pull && cd -
    fi
    
    echo "✅ SLAKE downloaded to $SLAKE_DIR"
}

# Main
echo "Which datasets would you like to download?"
echo "1) MIMIC-CXR (requires PhysioNet credentials)"
echo "2) VQA-RAD"
echo "3) SLAKE"
echo "4) All"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1) download_mimic_cxr ;;
    2) download_vqa_rad ;;
    3) download_slake ;;
    4) 
        download_mimic_cxr
        download_vqa_rad
        download_slake
        ;;
    *) echo "Invalid choice" ;;
esac

echo ""
echo "=== Download complete ==="
