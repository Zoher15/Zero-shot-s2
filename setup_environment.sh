#!/bin/bash

# Zero-shot-s² Environment Setup Script
# This script automates the environment setup process described in the README.md

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check if we're in a virtual environment
check_venv() {
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        return 0
    elif [[ "$CONDA_DEFAULT_ENV" != "" ]] && [[ "$CONDA_DEFAULT_ENV" != "base" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to verify CUDA availability
verify_cuda() {
    log "Verifying CUDA installation..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
        log_success "NVIDIA GPU detected"
    else
        log_warning "nvidia-smi not found. GPU acceleration may not be available."
    fi
}

# Function to verify PyTorch CUDA support
verify_pytorch_cuda() {
    log "Verifying PyTorch CUDA support..."
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version used by PyTorch: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA not available - will use CPU only')
"
}

# Function to create directory structure
create_directories() {
    log "Creating directory structure..."
    
    # Create data directories
    mkdir -p data/{d3,df40,genimage}
    
    # Create output directories
    mkdir -p outputs/{responses,scores,plots,tables}
    
    # Create cache and logs directories
    mkdir -p cache logs
    
    log_success "Directory structure created"
}

# Function to download NLTK data
download_nltk_data() {
    log "Downloading NLTK data..."
    python -c "
import nltk
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Warning: Could not download NLTK data: {e}')
"
}

# Main setup function
main() {
    log "Starting Zero-shot-s² environment setup..."
    
    # Check if we're in the correct directory
    if [[ ! -f "config.py" ]] || [[ ! -f "requirements.txt" ]]; then
        log_error "Please run this script from the Zero-shot-s² project root directory"
        exit 1
    fi
    
    # Verify CUDA setup
    verify_cuda
    
    # Check virtual environment
    if ! check_venv; then
        log_warning "No virtual environment detected!"
        log "Setting up virtual environment..."
        
        # Try conda first, then fallback to venv
        if check_conda; then
            log "Using conda to create virtual environment..."
            conda create -n zeroshot_s2 python=3.10 -y
            log_success "Conda environment 'zeroshot_s2' created"
            log "Please activate the environment with: conda activate zeroshot_s2"
            log "Then re-run this script."
            exit 0
        else
            log "Using venv to create virtual environment..."
            python -m venv venv
            log_success "Virtual environment created in ./venv"
            log "Please activate the environment with: source venv/bin/activate"
            log "Then re-run this script."
            exit 0
        fi
    else
        log_success "Virtual environment detected"
    fi
    
    # Create directory structure
    create_directories
    
    # Install dependencies in the correct order
    log "Installing dependencies..."
    
    # Step 1: Install PyTorch with CUDA 12.6 support
    log "Installing PyTorch with CUDA 12.6 support..."
    pip install --no-cache-dir \
        "torch==2.7.0+cu126" \
        "torchvision==0.22.0+cu126" \
        "torchaudio==2.7.0+cu126" \
        --extra-index-url https://download.pytorch.org/whl/test/cu126
    
    log_success "PyTorch installed"
    
    # Verify PyTorch installation
    verify_pytorch_cuda
    
    # Step 2: Install flash-attn dependencies
    log "Installing flash-attn dependencies..."
    pip install packaging ninja
    
    # Step 3: Install flash-attn (this can take a long time)
    log "Installing flash-attn (this may take 10-15 minutes or more)..."
    log_warning "Please be patient, flash-attn compilation can take a very long time..."
    pip install flash-attn --no-build-isolation --no-cache-dir
    
    log_success "flash-attn installed"
    
    # Step 4: Install remaining dependencies
    log "Installing remaining dependencies from requirements.txt..."
    pip install -r requirements.txt --no-cache-dir
    
    log_success "All dependencies installed"
    
    # Download NLTK data
    download_nltk_data
    
    # Final verification
    log "Running final verification..."
    verify_pytorch_cuda
    
    log_success "Environment setup complete!"
    log ""
    log "Next steps:"
    log "1. Prepare your datasets according to the README instructions"
    log "2. Place dataset CSV files and images in the data/ directories"
    log "3. Run experiments using scripts in the experiments/ directory"
    log ""
    log "For dataset preparation, refer to the 'Data Preparation' section in README.md"
    log ""
    log "Example commands to get started:"
    log "  # Download D3 dataset images"
    log "  python experiments/load_d3.py"
    log ""
    log "  # Run evaluation on GenImage dataset"
    log "  python experiments/evaluate_AI_qwen.py -llm qwen25-7b -c 0 -d genimage2k -b 20 -n 1 -m zeroshot-2-artifacts"
}

# Help function
show_help() {
    echo "Zero-shot-s² Environment Setup Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --verify-only  Only verify existing installation without installing"
    echo ""
    echo "This script will:"
    echo "1. Check for virtual environment (create one if needed)"
    echo "2. Create required directory structure"
    echo "3. Install PyTorch with CUDA 12.6 support"
    echo "4. Install flash-attn (takes time to compile)"
    echo "5. Install remaining dependencies from requirements.txt"
    echo "6. Download required NLTK data"
    echo "7. Verify the installation"
    echo ""
    echo "Prerequisites:"
    echo "- Python 3.10+ installed"
    echo "- NVIDIA GPU with CUDA drivers (optional but recommended)"
    echo "- Sufficient disk space for dependencies and compilation"
    echo ""
    echo "Notes:"
    echo "- Run this script from the Zero-shot-s² project root directory"
    echo "- If no virtual environment is active, the script will create one and exit"
    echo "- You'll need to activate the environment and re-run the script"
    echo "- The flash-attn installation can take 10-15 minutes or more"
}

# Verify-only function
verify_only() {
    log "Running verification checks only..."
    
    # Check if we're in the correct directory
    if [[ ! -f "config.py" ]] || [[ ! -f "requirements.txt" ]]; then
        log_error "Please run this script from the Zero-shot-s² project root directory"
        exit 1
    fi
    
    # Check virtual environment
    if ! check_venv; then
        log_error "No virtual environment detected!"
        exit 1
    else
        log_success "Virtual environment detected"
    fi
    
    # Verify CUDA
    verify_cuda
    
    # Verify PyTorch
    if python -c "import torch" 2>/dev/null; then
        verify_pytorch_cuda
        log_success "PyTorch is installed and working"
    else
        log_error "PyTorch is not installed"
        exit 1
    fi
    
    # Check flash-attn
    if python -c "import flash_attn" 2>/dev/null; then
        log_success "flash-attn is installed"
    else
        log_error "flash-attn is not installed"
        exit 1
    fi
    
    # Check other key dependencies
    log "Checking other key dependencies..."
    python -c "
import sys
dependencies = [
    'transformers', 'accelerate', 'huggingface_hub', 'pandas', 
    'numpy', 'scikit_learn', 'matplotlib', 'nltk', 'requests'
]
missing = []
for dep in dependencies:
    try:
        __import__(dep)
    except ImportError:
        missing.append(dep)

if missing:
    print(f'Missing dependencies: {missing}')
    sys.exit(1)
else:
    print('All key dependencies are installed')
"
    
    log_success "Environment verification complete!"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --verify-only)
        verify_only
        exit 0
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac 