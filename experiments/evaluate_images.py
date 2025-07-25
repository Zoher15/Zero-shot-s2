#!/usr/bin/env python3
"""
Unified evaluation script for AI-generated image detection using Vision-Language Models.

This script provides a unified interface for evaluating different VLM models on AI-generated 
image detection datasets. It follows Zero-shot-mod architecture patterns with modular design,
shared utilities, and comprehensive error handling.

Features:
    - Unified evaluation interface for Qwen, Llama, and CoDE models
    - Multiple prompting modes (zero-shot, zero-shot-cot, zero-shot-s²)
    - Self-consistency evaluation support
    - Adaptive batch sizing with OOM handling
    - Comprehensive logging and results output
    - GPU memory optimization
    - Two-stage generation process (reasoning + final answer)

Supported Models:
    - Qwen VLMs: qwen25-3b, qwen25-7b, qwen25-32b, qwen25-72b
    - Llama VLMs: llama3-11b, llama3-90b
    - CoDE: Classical computer vision model

Supported Datasets:
    - GenImage: genimage2k, genimage10k
    - DF40: df402k, df4010k  
    - D3: d32k

Usage:
    python experiments/evaluate_images.py -m zeroshot-2-artifacts -llm qwen25-7b -d df402k
    python experiments/evaluate_images.py -m zeroshot-cot -llm llama3-11b -d genimage2k -n 5
    python experiments/evaluate_images.py -m zeroshot -llm code -d d32k
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def parse_args_early():
    """Parse arguments early to set up CUDA environment before importing PyTorch."""
    parser = argparse.ArgumentParser(description="Unified AI-generated image detection evaluation script")
    parser.add_argument("-m", "--mode", type=str, 
                       help="Mode of reasoning", 
                       choices=["zeroshot", "zeroshot-cot", "zeroshot-2-artifacts", "zeroshot-3-artifacts", 
                               "zeroshot-4-artifacts", "zeroshot-5-artifacts", "zeroshot-6-artifacts",
                               "zeroshot-7-artifacts", "zeroshot-8-artifacts", "zeroshot-9-artifacts"], 
                       default="zeroshot-2-artifacts")
    parser.add_argument("-llm", "--llm", type=str, 
                       help="The name of the model", 
                       choices=["qwen25-3b", "qwen25-7b", "qwen25-32b", "qwen25-72b",
                               "llama3-11b", "llama3-90b", "code"],
                       default="qwen25-7b")
    parser.add_argument("-c", "--cuda", type=str, 
                       help="CUDA devices", 
                       default="0")
    parser.add_argument("-d", "--dataset", type=str, 
                       help="Dataset to evaluate", 
                       choices=["genimage2k", "genimage10k", "df402k", "df4010k", "d32k"], 
                       default="df402k")
    parser.add_argument("-b", "--batch_size", type=int, 
                       help="Batch size for inference", 
                       default=20)
    parser.add_argument("-n", "--num_sequences", type=int,
                       help="Number of sequences for self-consistency",
                       default=1)
    parser.add_argument("--max_new_tokens", type=int,
                       help="Maximum new tokens to generate",
                       default=300)
    
    return parser.parse_args()

# Parse arguments and set up CUDA BEFORE importing PyTorch
args = parse_args_early()

# Now import config and setup CUDA environment
import config
from utils.helpers import (
    initialize_environment, setup_global_logger, initialize_evaluation_state,
    load_test_data_unified, save_evaluation_outputs, validate_model_kwargs,
    clear_gpu_memory, get_generation_kwargs, validate_answers, update_progress,
    get_macro_f1_from_counts, run_model_evaluation, setup_evaluation_environment,
    load_vlm_model_unified, create_vlm_response_generator_unified
)

# Initialize environment early
initialize_environment(args.cuda)

# Import PyTorch after CUDA setup
import torch
from tqdm import tqdm

# Setup logger
logger = logging.getLogger(__name__)


class CudaOOMError(Exception):
    """Custom exception for CUDA out of memory errors"""
    pass


def parse_arguments():
    """Return the pre-parsed arguments."""
    return args


# All VLM model loading and response generation functions are now in helpers.py
# This maintains the modular architecture following Zero-shot-mod patterns


# All response generation functions are now in helpers.py
# This maintains the modular architecture following Zero-shot-mod patterns


def main():
    """Main execution function."""
    try:
        args = parse_arguments()
        
        # Validate arguments
        if args.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if args.num_sequences <= 0:
            raise ValueError("num_sequences must be positive")
        if args.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        
        # Setup comprehensive evaluation environment
        log_file = config.get_model_output_dir(args.llm, 'logs') / config.get_filename(
            "log", args.dataset, args.llm, args.mode, args.num_sequences
        )
        setup_evaluation_environment(args.cuda, log_file, args.llm)
        
        logger.info(f"Starting evaluation with arguments: {vars(args)}")
        
        # Load test data
        question_phrase = config.EVAL_QUESTION_PHRASE
        test_data = load_test_data_unified(args.dataset, question_phrase, config)
        
        if not test_data:
            raise ValueError(f"Failed to load test data for dataset: {args.dataset}")
        
        logger.info(f"Successfully loaded {len(test_data)} examples for dataset '{args.dataset}'")
        
        # Load VLM model
        logger.info(f"Loading model: {args.llm}")
        try:
            model, processor, model_type = load_vlm_model_unified(args.llm)
            logger.info(f"Successfully loaded {model_type} model: {args.llm}")
        except Exception as e:
            logger.error(f"Failed to load model {args.llm}: {e}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {e}")
        
        # Create response generator
        response_generator = create_vlm_response_generator_unified(model, processor, model_type, args.mode)
        
        # Determine model prefix for output files
        model_prefix_map = {
            'qwen': 'AI_qwen',
            'llama': 'AI_llama', 
            'code': 'AI_CoDE'
        }
        model_prefix = model_prefix_map.get(model_type, f"AI_{model_type}")
        
        # Run evaluation using shared logic
        logger.info("Starting evaluation...")
        print(f"Starting evaluation: {args.llm} on {args.dataset} dataset")
        print(f"Mode: {args.mode}, Sequences: {args.num_sequences}")
        
        try:
            final_f1_score = run_model_evaluation(
                test_data=test_data,
                response_generator_fn=response_generator,
                model_name=args.llm,
                mode_type=args.mode,
                num_sequences=args.num_sequences,
                batch_size=args.batch_size,
                dataset_name=args.dataset,
                model_prefix=model_prefix,
                config_module=config
            )
            
            logger.info(f"Evaluation completed successfully!")
            logger.info(f"Model: {args.llm}")
            logger.info(f"Dataset: {args.dataset}")
            logger.info(f"Mode: {args.mode}")
            logger.info(f"Sequences: {args.num_sequences}")
            logger.info(f"Final Macro F1-Score: {final_f1_score:.4f}")
            
            # Print final results
            print(f"\nFinal Results:")
            print(f"Macro F1-Score: {final_f1_score:.4f}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            print(f"Error: {e}")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())