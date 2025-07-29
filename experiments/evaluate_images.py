#!/usr/bin/env python3
"""
Unified evaluation script for Zero-shot-s² AI-generated image detection.

This script provides a unified interface for evaluating vision-language models and 
classification models on AI-generated image detection tasks. It supports multiple 
model families and prompting modes for comprehensive evaluation.

Features:
    - Vision-language model evaluation (Qwen, Llama)
    - Direct classification model evaluation (CoDE)
    - Multiple prompting modes (zero-shot, chain-of-thought, zero-shot-s²)
    - Self-consistency evaluation support
    - Comprehensive logging and results output
    - GPU memory optimization with CUDA OOM handling
    - Two-stage generation process for VLMs (reasoning + final answer)

Supported Models:
    - Qwen2.5-VL: qwen25-3b, qwen25-7b, qwen25-32b, qwen25-72b
    - Llama Vision: llama3-11b, llama3-90b
    - CoDE: code (direct classification)

Supported Datasets:
    - GenImage: genimage2k, genimage10k
    - DF40: df402k, df4010k  
    - D3: d32k

Usage:
    python experiments/evaluate_images.py -llm qwen25-7b -m zeroshot-2-artifacts -d df402k -c 0
    python experiments/evaluate_images.py -llm llama3-11b -m zeroshot-cot -d genimage2k -n 5
    python experiments/evaluate_images.py -llm code -d d32k -b 50
"""

import argparse
import concurrent.futures
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def parse_args_early():
    """Parse arguments early to set up CUDA environment before importing PyTorch."""
    parser = argparse.ArgumentParser(description="Unified Zero-shot-s² Image Evaluation Script")
    
    # Model selection
    parser.add_argument("-llm", "--llm", type=str, required=True,
                       help="Model to evaluate",
                       choices=["qwen25-3b", "qwen25-7b", "qwen25-32b", "qwen25-72b",
                               "llama3-11b", "llama3-90b", "code", "o3"])
    
    # Evaluation parameters
    parser.add_argument("-m", "--mode", type=str, default="zeroshot-2-artifacts",
                       help="Prompting mode (default: zeroshot-2-artifacts)",
                       choices=["zeroshot", "zeroshot-cot", "zeroshot-2-artifacts", "zeroshot-3-artifacts", 
                               "zeroshot-4-artifacts", "zeroshot-5-artifacts", "zeroshot-6-artifacts",
                               "zeroshot-7-artifacts", "zeroshot-8-artifacts", "zeroshot-9-artifacts", 
                               "direct_classification"])
    parser.add_argument("-d", "--dataset", type=str, default="df402k",
                       help="Dataset to evaluate (default: df402k)",
                       choices=["genimage2k", "genimage10k", "df402k", "df4010k", "d32k"])
    parser.add_argument("-c", "--cuda", type=str, default="0",
                       help="CUDA device IDs (default: 0)")
    parser.add_argument("-b", "--batch_size", type=int, default=20,
                       help="Batch size for inference (default: 20)")
    parser.add_argument("-n", "--num", type=int, default=1,
                       help="Number of sequences for self-consistency (default: 1)")
    
    return parser.parse_args()

# Parse arguments and initialize environment BEFORE PyTorch imports
args = parse_args_early()

# Import config and helpers after argument parsing
import config
from utils import helpers
from openai_requests import get_openai_responses

# Initialize environment and logging with individual log files
helpers.initialize_environment(args.cuda)

# Create individual log file for this specific evaluation run using model-organized structure
log_file = config.get_model_output_dir(args.llm, 'logs', args.mode) / config.get_filename(
    "log", dataset=args.dataset, model=args.llm, mode=args.mode, num_seq=args.num
)
helpers.setup_global_logger(log_file)
logger = logging.getLogger(__name__)

# NOW import PyTorch and model-specific libraries
import torch
import random
from PIL import Image
import requests

# Model-specific imports (conditional to avoid unnecessary dependencies)
if args.llm.startswith('qwen'):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        logger.error("qwen_vl_utils not available. Please install qwen-vl-utils for Qwen model support.")
        sys.exit(1)
elif args.llm.startswith('llama'):
    from transformers import MllamaForConditionalGeneration, AutoProcessor
elif args.llm == 'code':
    import torch.nn as nn
    from torchvision import transforms
    import joblib
    import transformers
    from huggingface_hub import hf_hub_download
elif args.llm == 'o3':
    import os
    import time
    import json
    import tempfile
    from openai import OpenAI


# =============================================================================
# BATCH PROCESSING WITH OOM HANDLING
# =============================================================================

def process_model_batch(model, processor, prompts, model_kwargs, extra_args, model_name, batch_examples_expanded):
    """
    Process a batch of prompts through the model with unified image handling.
    
    Args:
        model: The loaded model for text generation
        processor: Model processor 
        prompts: List of prompt strings to process
        model_kwargs: Dictionary of model generation parameters
        extra_args: Additional arguments for generation (return_dict_in_generate, output_scores, etc.)
        model_name: Name of the model (to determine processing approach)
        batch_examples_expanded: List of examples with image and message data
        
    Returns:
        List of generated response strings
        
    Raises:
        helpers.CudaOOMError: If CUDA out of memory error occurs
        Exception: For other model processing errors
    """
    try:
        # Process images based on model type and tokenize
        if model_name.startswith('qwen'):
            # Qwen: Extract images from messages and pass as flat list
            all_image_inputs = []
            all_video_inputs = []
            
            for example in batch_examples_expanded:
                messages = example.get('messages', [])
                img_inputs, vid_inputs = process_vision_info(messages)
                if img_inputs:
                    all_image_inputs.extend(img_inputs)
                if vid_inputs:
                    all_video_inputs.extend(vid_inputs)
            
            # Qwen processor call
            inputs = processor(
                text=prompts, 
                images=all_image_inputs if all_image_inputs else None, 
                videos=all_video_inputs if all_video_inputs else None, 
                padding=True, 
                return_tensors="pt"
            ).to(model.device)
            
        elif model_name.startswith('llama'):
            # Llama: Use PIL Image.open from file paths, nested list format
            image_inputs = [[Image.open(example['image']).convert("RGB")] for example in batch_examples_expanded]
            
            # Processor call
            inputs = processor(
                text=prompts, 
                images=image_inputs, 
                padding=True, 
                return_tensors="pt"
            ).to(model.device)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        input_length = inputs.input_ids.shape[1]
        
        # Generate responses with unified extra args
        merged_args = {**inputs, **model_kwargs, **extra_args}
        
        with torch.no_grad():
            outputs = model.generate(**merged_args)
        
        # Decode responses
        responses = processor.batch_decode(
            outputs.sequences[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # Memory cleanup
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return responses
        
    except Exception as e:
        # Check if it's a CUDA OOM error specifically
        if "CUDA out of memory" in str(e) or "OutOfMemoryError" in str(e):
            logger.error(f"CUDA out of memory during model processing: {e}")
            raise helpers.CudaOOMError(f"CUDA out of memory: {e}")
        else:
            logger.error(f"Error in model processing: {e}")
            raise

# =============================================================================
# MODEL LOADING FUNCTIONS (FUNCTIONAL APPROACH)
# =============================================================================

def load_qwen_model(model_name: str):
    """Load Qwen model and processor."""
    model_mapping = {
        "qwen25-3b": "Qwen/Qwen2.5-VL-3B-Instruct", 
        "qwen25-7b": "Qwen/Qwen2.5-VL-7B-Instruct", 
        "qwen25-32b": "Qwen/Qwen2.5-VL-32B-Instruct", 
        "qwen25-72b": "Qwen/Qwen2.5-VL-72B-Instruct"
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported Qwen model: {model_name}")
    
    model_path = model_mapping[model_name]
    
    logger.info(f"Loading Qwen model: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", 
        attn_implementation="flash_attention_2", use_cache=True
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    # Set padding to left for generation tasks
    processor.tokenizer.padding_side = "left"
    
    return model, processor

def load_llama_model(model_name: str):
    """Load Llama model and processor."""
    model_mapping = {
        "llama3-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "llama3-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported Llama model: {model_name}")
    
    model_path = model_mapping[model_name]
    
    logger.info(f"Loading Llama model: {model_path}")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    return model, processor

def load_code_model():
    """Load CoDE model."""
    import torch.nn as nn
    from torchvision import transforms
    import joblib
    import transformers
    from huggingface_hub import hf_hub_download
    
    class VITContrastiveHF(nn.Module):
        def __init__(self, repo_name='aimagelab/CoDE', cache_dir=None):
            super(VITContrastiveHF, self).__init__()
            
            self.model = transformers.AutoModel.from_pretrained(repo_name, cache_dir=cache_dir)
            self.model.pooler = nn.Identity()
            
            self.processor = transformers.AutoProcessor.from_pretrained(repo_name, cache_dir=cache_dir)
            self.processor.do_resize = False
            
            file_path = hf_hub_download(
                repo_id=repo_name, 
                filename='sklearn/linear_tot_classifier_epoch-32.sav', 
                cache_dir=cache_dir
            )
            self.classifier = joblib.load(file_path)

        def forward(self, x, return_feature=False):
            features = self.model(x)
            
            if return_feature:
                return features
                
            features = features.last_hidden_state[:, 0, :].cpu().detach().numpy()
            predictions = self.classifier.predict(features)
            
            return torch.from_numpy(predictions)
    
    logger.info("Loading CoDE model from Hugging Face...")
    model = VITContrastiveHF(repo_name='aimagelab/CoDE').eval().to(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    logger.info("CoDE model loaded successfully.")
    return model, None

# =============================================================================
# PROMPT BUILDING FUNCTIONS
# =============================================================================

def build_prompts(examples, mode_type, processor=None, model_name=None):
    """
    Build prompts for different model types with explicit format control.
    
    Args:
        examples: List of example dictionaries with 'image' and 'question' keys
        mode_type: Prompting mode (zeroshot, zeroshot-cot, etc.)
        processor: Model processor for HuggingFace models
        model_name: Model name to determine format type ("qwen25-7b", "llama3-11b", etc.)
        
    
    Note: 
        Modifies examples in-place by adding 'prompt' key to each example.
        For Qwen models, also adds 'messages' key for use with process_vision_info.
        No return value needed since examples are modified directly.
    """
    instructions_text = helpers.get_instructions_for_mode(mode_type)
    
    # Extract format type from model name
    if model_name is None:
        raise ValueError("model_name is required to determine format type.")
    
    if model_name.startswith('qwen'):
        format_type = "qwen"
    elif model_name.startswith('llama'):
        format_type = "llama"
    elif model_name == 'o3':
        format_type = "openai"
    else:
        raise ValueError(f"Unknown model format for: {model_name}")
    
    # Common format for all three: system message + user message structure
    for example in examples:
        messages = []
        # Build system message if instructions exist (common)
        if instructions_text != "":
            messages.append({"role": "system", "content": instructions_text})
        
        # Build user message - content differs by format
        if format_type == "llama":
            # Llama: generic image type
            user_content = [
                {"type": "image"},
                {"type": "text", "text": example['question']}
            ]
        elif format_type == "qwen":
            # Qwen: specific image path
            user_content = [
                {"type": "image", "image": example['image']},
                {"type": "text", "text": example['question']}
            ]
        elif format_type == "openai":
            # OpenAI: base64 encoded image
            import base64
            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
            
            base64_image = encode_image(example['image'])
            user_content = [
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                {"type": "input_text", "text": example['question']}
            ]
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Store original messages for all formats (consistent structure)
        example['messages'] = messages
        
        # Apply chat template or handle messages - differs by format
        if format_type == "openai":
            # OpenAI: Add reasoning prefix to system message if needed
            if mode_type in config.REASONING_PREFIXES:
                reasoning_prefix = config.REASONING_PREFIXES[mode_type]
                # Create system message with reasoning prefix instruction
                messages.insert(0, {
                    "role": "system", 
                    "content": f"Start your responses with \"{reasoning_prefix}\""
                })
            
            example['prompt'] = "\n".join([f"{msg['role'].title()}: {msg['content'] if isinstance(msg['content'], str) else next((item['text'] for item in msg['content'] if item.get('type') in ['input_text', 'text']), '')}" for msg in messages])
        else:
            # Llama/Qwen: use chat template to create string prompt
            chat_template_kwargs = {"add_generation_prompt": True}
            if format_type == "qwen":
                chat_template_kwargs["tokenize"] = False
                
            prompt_text = processor.apply_chat_template(
                messages, **chat_template_kwargs
            )
            
            # Add reasoning prefix if needed
            if mode_type in config.REASONING_PREFIXES:
                final_text = f"{prompt_text}{config.REASONING_PREFIXES[mode_type]}"
            else:
                final_text = prompt_text
            example['prompt'] = final_text

# =============================================================================
# RESPONSE GENERATION FUNCTIONS
# =============================================================================

def get_responses(model, processor, batch_examples, model_name, model_kwargs, answer_phrase):
    """
    Generate responses for both Qwen and Llama models with unified two-stage generation.
    
    This function implements a two-stage generation process using pre-built prompts:
    1. First generation: Generate reasoning/analysis based on the prompt
    2. Second generation: Generate the final answer after the reasoning
    
    Args:
        model: The loaded model for text generation
        processor: Model processor 
        batch_examples: List of example dictionaries with 'prompt', 'image', and (for Qwen) 'messages' keys
        model_name: Name of the model (to determine processing approach)
        model_kwargs: Dictionary of model generation parameters
        answer_phrase: Phrase that separates reasoning from final answer
        
    Returns:
        List of complete response strings combining reasoning and final responses
    """
    try:
        model_kwargs_copy = model_kwargs.copy()
        
        # Handle num_return_sequences by duplicating prompts and examples
        if 'top_k_beams' in model_kwargs_copy:
            k = model_kwargs_copy.pop('top_k_beams')
        elif 'num_return_sequences' in model_kwargs_copy:
            k = model_kwargs_copy.pop('num_return_sequences')
        else:
            k = 1

        # Prepare examples for self-consistency
        if k > 1:
            # Self-consistency mode: duplicate each example k times
            batch_examples_expanded = []
            for example in batch_examples:
                batch_examples_expanded.extend([example] * k)
        else:
            # Regular batch mode
            batch_examples_expanded = batch_examples
        
        # Extract prompts from expanded examples
        first_prompts = [example['prompt'] for example in batch_examples_expanded]
        
        # ===== FIRST STAGE: REASONING GENERATION =====
        
        # Unified extra args
        extra_args = {
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True,
        }
        
        # First generation
        first_responses = process_model_batch(
            model, processor, first_prompts, 
            model_kwargs_copy, extra_args, model_name, 
            batch_examples_expanded
        )
        
        # ===== SECOND STAGE: ANSWER GENERATION =====
        
        # Prepare second generation prompts (like zero_shot_mod)
        second_prompts = [f"{example['prompt']}{first_response}\n\n{answer_phrase}" for example, first_response in zip(batch_examples_expanded, first_responses)]
        
        # Second generation with reduced max tokens for faster inference  
        model_kwargs_copy['max_new_tokens'] = 50
        
        # Second generation
        second_responses = process_model_batch(
            model, processor, second_prompts, 
            model_kwargs_copy, extra_args, model_name, 
            batch_examples_expanded
        )
        
        # Combine first and second responses (like zero_shot_mod)
        full_responses = [f"{first_response}\n\n{answer_phrase}{second_response}" for first_response, second_response in zip(first_responses, second_responses)]
        
        return full_responses
        
    except Exception as e:
        # Check if it's a CUDA OOM error specifically
        if "CUDA out of memory" in str(e) or "OutOfMemoryError" in str(e):
            logger.error(f"CUDA OOM during response generation for {model_name}: {e}")
            raise helpers.CudaOOMError(f"CUDA out of memory: {e}")
        else:
            logger.error(f"Error in model processing: {e}")
            raise


def generate_code_responses(model, processor, batch_examples, mode_type, model_kwargs, answer_phrase=None):
    """Generate responses using CoDE direct classification."""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    batch_tensors = []
    image_paths = [ex['image'] for ex in batch_examples]
    
    # Process images
    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert('RGB')
            tensor = transform(img)
            batch_tensors.append(tensor)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}. Skipping.")
    
    if not batch_tensors:
        return []
    
    # Stack and move to device
    input_batch = torch.stack(batch_tensors).to(model.model.device)
    
    # Generate predictions
    responses = []
    with torch.no_grad():
        predictions = model(input_batch).cpu().tolist()
    
    # Convert predictions to text
    for pred in predictions:
        if pred == 1:
            responses.append('ai-generated')
        elif pred == 0 or pred == -1:
            responses.append('real')
        else:
            logger.warning(f"Unknown prediction value: {pred}")
            responses.append('unknown')
    
    return responses

# =============================================================================
# UNIFIED EVALUATION FUNCTION
# =============================================================================

def evaluate_model(test_data, model, processor, model_name, mode_type, model_kwargs, batch_size=20, dataset_name='images'):
    """
    Unified model evaluation function for AI-generated image detection.
    
    Evaluates model performance on image detection tasks by processing examples in batches,
    generating responses, and computing macro F1 scores. Supports HuggingFace models 
    (Qwen, Llama VLMs, CoDE) and OpenAI models (O3).
    
    Args:
        test_data: List of test examples to evaluate
        model: The loaded model for inference (None for OpenAI models)
        processor: Model processor (for VLMs) or None (for CoDE/OpenAI)
        model_name: Name of the model (determines processing approach)
        mode_type: String specifying the reasoning mode
        model_kwargs: Dictionary of model generation parameters
        batch_size: Number of examples to process in each batch
        
    Returns:
        tuple: (score, responses_data, skipped_data) where:
            - score: Dictionary containing macro F1 scores and metrics
            - responses_data: List of response entries with predictions and ground truth
            - skipped_data: List of examples that failed to process
    """
    from tqdm import tqdm
    
    # Initialize evaluation state
    score = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'correct': 0, 'n': 0, 'macro_f1': 0.0}
    responses_data = []
    skipped_data = []
    
    # Build ALL prompts at once for length-based sorting optimization
    logger.info("Building prompts for all examples...")
    build_prompts(test_data, mode_type, processor=processor, model_name=model_name)
    
    # Sort by image size (largest first) to minimize padding waste in batches
    # Skip sorting for OpenAI models to avoid clustering large images in first batch
    if model_name != 'o3':
        logger.info("Sorting examples by image size for efficient batching...")
        def get_image_pixel_count(example):
            """Get pixel count without loading full image data (fast metadata read)."""
            try:
                with Image.open(example['image']) as img:
                    return img.width * img.height
            except Exception:
                return 0  # Fallback for any file issues
        
        test_data.sort(key=get_image_pixel_count, reverse=True)  # Sort by image size
    else:
        logger.info("Skipping sorting for OpenAI model to ensure balanced batch sizes")
    
    # Progress bar
    pbar = tqdm(total=len(test_data), desc="Evaluating")
    current_index = 0

    # Calculate effective batch size
    num_return_sequences = model_kwargs.get('num_return_sequences', 1)
    if model_name == 'o3':
        # O3: Process all examples in single batch, but handle chunking internally
        effective_batch_size = len(test_data)
    elif batch_size > num_return_sequences:
        effective_batch_size = batch_size // num_return_sequences
    else:
        effective_batch_size = num_return_sequences

    # Process examples in batches using sorted data
    while current_index < len(test_data):
        batch_start = current_index
        batch_end = min(batch_start + effective_batch_size, len(test_data))
        batch_examples = test_data[batch_start:batch_end]
        
        try:
            # Generate responses based on model type using pre-built prompts
            if model_name.startswith(('qwen', 'llama')):
                responses = get_responses(
                    model, processor, batch_examples, model_name, model_kwargs, config.EVAL_ANSWER_PHRASE
                )
            elif model_name == 'code':
                responses = generate_code_responses(
                    model, processor, batch_examples, mode_type, model_kwargs
                )
            elif model_name == 'o3':
                responses = get_openai_responses(
                    batch_examples, model_name, model_kwargs, mode_type, dataset_name
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
                
        except Exception as e:
            logger.error(f"Error processing batch starting at {batch_start}: {e}")
            for i, example in enumerate(batch_examples):
                skipped_data.append({'example': example, 'error': str(e), 'index': batch_start + i})
            # Still update progress bar for skipped examples
            pbar.update(len(batch_examples))
            current_index = batch_end
            continue
        
        # Handle failed examples
        if responses is None:
            # Non-CUDA error - treat all as failed
            failed_indices = list(range(len(batch_examples)))
        else:
            failed_indices = []
        
        if failed_indices:
            success_count = len(responses) - len(failed_indices)
            logger.warning(f"Batch starting at {batch_start}: {len(failed_indices)} examples failed, {success_count} succeeded")
            for failed_idx in failed_indices:
                example = batch_examples[failed_idx]
                skipped_data.append({'example': example, 'error': 'Model processing failed (non-CUDA error)', 'index': batch_start + failed_idx})
        
        # If all examples failed, continue to next batch
        if len(failed_indices) == len(batch_examples):
            pbar.update(len(batch_examples))
            current_index = batch_end
            continue
        
        # Process results using shared helper
        score, responses_data, skipped_data = helpers.process_evaluation_batch(
            batch_examples, responses, score, responses_data, skipped_data, pbar,
            failed_indices=failed_indices, batch_start_idx=batch_start, 
            num_return_sequences=num_return_sequences, model_name=model_name,
            answer_phrase=config.EVAL_ANSWER_PHRASE
        )
        
        # Memory cleanup
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        current_index = batch_end
    
    pbar.close()
    
    # Final macro F1 score is already updated via rolling calculation
    return score, responses_data, skipped_data



def main():
    """Main execution function."""
    logger.info(f"Starting unified evaluation with model: {args.llm}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate configuration
    try:
        config.validate_dataset_mode(args.dataset, args.mode, args.llm)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Load test data
    question_phrase = config.EVAL_QUESTION_PHRASE
    test_data = helpers.load_dataset_examples(args.dataset, question_phrase, config)
    
    if not test_data:
        logger.error(f"Failed to load test data for dataset: {args.dataset}")
        sys.exit(1)
    
    logger.info(f"Successfully loaded {len(test_data)} examples for dataset '{args.dataset}'")
    
    # Load model
    try:
        if args.llm.startswith('qwen'):
            model, processor = load_qwen_model(args.llm)
        elif args.llm.startswith('llama'):
            model, processor = load_llama_model(args.llm)
        elif args.llm == 'code':
            model, processor = load_code_model()
        elif args.llm == 'o3':
            model, processor = None, None  # O3 doesn't need local model loading
        else:
            raise ValueError(f"Unsupported model: {args.llm}")
        
        logger.info(f"Successfully loaded model: {args.llm}")
    except Exception as e:
        logger.error(f"Failed to load model {args.llm}: {e}", exc_info=True)
        sys.exit(1)
    
    # Get generation parameters
    model_kwargs = config.get_generation_kwargs(args.num)
    config.validate_model_kwargs(model_kwargs)
    
    # Run evaluation using unified function
    try:
        score, responses_data, skipped_data = evaluate_model(
            test_data=test_data,
            model=model,
            processor=processor,  
            model_name=args.llm,
            mode_type=args.mode,
            model_kwargs=model_kwargs,
            batch_size=args.batch_size,
            dataset_name=args.dataset
        )
        
        final_f1_score = score['macro_f1']
        
        # Save evaluation outputs
        helpers.save_evaluation_outputs(
            rationales_data=responses_data,
            score_metrics=score,
            macro_f1_score=final_f1_score,
            dataset_name=args.dataset,
            model_string=args.llm,
            mode_type_str=args.mode,
            num_sequences_val=args.num,
            config_module=config,
            skipped_data=skipped_data
        )
        
        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Model: {args.llm}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Sequences: {args.num}")
        logger.info(f"Final Macro F1-Score: {final_f1_score:.4f}")
        
        # Print final results for user
        print(f"\n🎯 Evaluation Results:")
        print(f"Model: {args.llm}")
        print(f"Dataset: {args.dataset}")  
        print(f"Mode: {args.mode}")
        print(f"Macro F1-Score: {final_f1_score:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()