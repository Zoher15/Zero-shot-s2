"""
Qwen Vision-Language Model Evaluation Script

This script evaluates Qwen vision-language models for AI-generated image detection
using the Zero-shot-s² framework. It implements a clean functional approach with
shared evaluation logic from helpers.py to eliminate code duplication.

The script uses Qwen2VLForConditionalGeneration from Hugging Face and implements
a two-stage generation process to ensure proper answer formatting. It supports
all prompting modes and self-consistency evaluation.

Features:
- Clean functional design without classes
- Two-stage generation for proper answer formatting
- All prompting modes: zero-shot, zero-shot-cot, zero-shot-s²
- Self-consistency with multiple response sampling
- Memory-efficient processing with GPU cache management
- Shared evaluation logic with other model scripts

Supported Models:
- qwen25-7b: Qwen2.5-VL 7B Instruct
- qwen25-3b: Qwen2.5-VL 3B Instruct
- qwen25-32b: Qwen2.5-VL 32B Instruct
- qwen25-72b: Qwen2.5-VL 72B Instruct

Usage:
    python experiments/evaluate_AI_qwen.py [options]
    
Examples:
    # Basic zero-shot-s² evaluation
    python experiments/evaluate_AI_qwen.py -m zeroshot-2-artifacts -d df402k
    
    # Self-consistency evaluation with 5 samples
    python experiments/evaluate_AI_qwen.py -m zeroshot-2-artifacts -d df402k -n 5
    
    # Chain-of-thought evaluation
    python experiments/evaluate_AI_qwen.py -m zeroshot-cot -d genimage2k -b 10

Command Line Arguments:
    -m, --mode: Prompting mode (default: zeroshot-2-artifacts)
    -llm, --llm: Qwen model name (default: qwen25-7b)
    -c, --cuda: CUDA device IDs (default: 7)
    -d, --dataset: Dataset to evaluate (default: df402k)
    -b, --batch_size: Batch size for inference (default: 20)
    -n, --num: Number of sequences for self-consistency (default: 1)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import argparse

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import config
from utils import helpers

# =============================================================================
# EARLY ARGUMENT PARSING FOR CUDA SETUP
# =============================================================================

def parse_args_early():
    """Parse arguments early to set CUDA environment before PyTorch import."""
    parser = argparse.ArgumentParser(description="Qwen Vision-Language Model Evaluation Script")
    parser.add_argument("-m", "--mode", type=str, default="zeroshot-2-artifacts",
                       help="Prompting mode (default: zeroshot-2-artifacts)")
    parser.add_argument("-llm", "--llm", type=str, default="qwen25-7b",
                       help="Qwen model name (default: qwen25-7b)")
    parser.add_argument("-c", "--cuda", type=str, default="7",
                       help="CUDA device IDs (default: 7)")
    parser.add_argument("-d", "--dataset", type=str, default="df402k",
                       help="Dataset to evaluate (default: df402k)")
    parser.add_argument("-b", "--batch_size", type=int, default=20,
                       help="Batch size for inference (default: 20)")
    parser.add_argument("-n", "--num", type=int, default=1,
                       help="Number of sequences for self-consistency (default: 1)")
    
    return parser.parse_args()

# Parse arguments and initialize environment BEFORE PyTorch imports
args = parse_args_early()
helpers.initialize_environment(args.cuda)

# Set up logging early
helpers.setup_global_logger(config.EVAL_QWEN_LOG_FILE)
logger = logging.getLogger(__name__)

# NOW it's safe to import PyTorch and transformers
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import random

# =============================================================================
# QWEN MODEL CONFIGURATION
# =============================================================================

# Supported Qwen models
QWEN_MODELS = {
    "qwen25-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen25-3b": "Qwen/Qwen2.5-VL-3B-Instruct", 
    "qwen25-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen25-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
}

# =============================================================================
# VLM MODEL FUNCTIONS ARE NOW IN HELPERS.PY
# =============================================================================

# All VLM model loading and response generation functions have been moved to 
# utils/helpers.py to follow Zero-shot-mod's modular architecture patterns.
# This eliminates code duplication and provides a unified interface for all 
# vision-language models (Qwen, Llama, CoDE).
#
# The unified functions used are:
# - helpers.load_vlm_model_unified() - Loads any supported VLM model
# - helpers.create_vlm_response_generator_unified() - Creates response generators

# Duplicate functions have been removed. All VLM functionality is now in helpers.py

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    logger.info(f"Starting Qwen evaluation with model: {args.llm}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate model name
    if args.llm not in QWEN_MODELS:
        available_models = list(QWEN_MODELS.keys())
        logger.error(f"Unsupported model: {args.llm}. Available models: {available_models}")
        sys.exit(1)
    
    # Load test data
    question_phrase = config.EVAL_QUESTION_PHRASE
    test_data = helpers.load_test_data_unified(args.dataset, question_phrase, config)
    
    if not test_data:
        logger.error(f"Failed to load test data for dataset: {args.dataset}")
        sys.exit(1)
    
    logger.info(f"Successfully loaded {len(test_data)} examples for dataset '{args.dataset}'")
    
    # Load Qwen model using unified architecture
    try:
        model, processor, model_type = helpers.load_vlm_model_unified(args.llm)
        logger.info(f"Successfully loaded {model_type} model: {args.llm}")
    except Exception as e:
        logger.error(f"Failed to load model {args.llm}: {e}", exc_info=True)
        sys.exit(1)
    
    # Create response generator using unified architecture
    response_generator = helpers.create_vlm_response_generator_unified(model, processor, model_type, args.mode)
    
    # Run evaluation using shared logic
    try:
        final_f1_score = helpers.run_model_evaluation(
            test_data=test_data,
            response_generator_fn=response_generator,
            model_name=args.llm,
            mode_type=args.mode,
            num_sequences=args.num,
            batch_size=args.batch_size,
            dataset_name=args.dataset,
            model_prefix="AI_qwen",
            config_module=config
        )
        
        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Model: {args.llm}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Sequences: {args.num}")
        logger.info(f"Final Macro F1-Score: {final_f1_score:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()