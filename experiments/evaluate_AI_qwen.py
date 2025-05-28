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
# QWEN MODEL FUNCTIONS
# =============================================================================

def load_qwen_model(model_name: str) -> Tuple[Any, Any]:
    """
    Load Qwen vision-language model and processor.
    
    Args:
        model_name: Name of the Qwen model
        
    Returns:
        Tuple of (model, processor)
        
    Raises:
        ValueError: If model name is not supported
    """
    if model_name not in QWEN_MODELS:
        available_models = list(QWEN_MODELS.keys())
        raise ValueError(f"Unsupported Qwen model: {model_name}. Available: {available_models}")
    
    model_path = QWEN_MODELS[model_name]
    
    logger.info(f"Loading Qwen processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    
    logger.info(f"Loading Qwen model: {model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    
    logger.info("Compiling Qwen model...")
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        logger.info("Qwen model compilation complete.")
    except Exception as e:
        logger.warning(f"Qwen model compilation failed: {e}. Proceeding without compilation.")
    
    return model, processor

def prepare_qwen_messages(examples: List[Dict]) -> List[List[Dict]]:
    """
    Prepare message format for Qwen models.
    
    Args:
        examples: List of example dictionaries with 'image' and 'question' keys
        
    Returns:
        List of message lists in Qwen format
    """
    messages_list = []
    for example in examples:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": example['image']},
                {"type": "text", "text": example['question']}
            ]
        }]
        messages_list.append(messages)
    return messages_list

def prepare_qwen_prompts(examples: List[Dict], mode_type: str) -> List[str]:
    """
    Prepare prompts for Qwen evaluation.
    
    Args:
        examples: List of example dictionaries
        mode_type: Prompting mode
        
    Returns:
        List of formatted prompt strings
    """
    instructions_text = helpers.get_instructions_for_mode(mode_type)
    prompt_texts = []
    
    for example in examples:
        # Qwen uses simple text format
        prompt_text = example['question']
        if instructions_text:
            prompt_text = f"{instructions_text}\n\n{prompt_text}"
        
        # Add mode-specific prefix
        final_prompt_text = helpers.get_model_guiding_prefix_for_mode(prompt_text, mode_type)
        prompt_texts.append(final_prompt_text)
        
        # Store prompt in example for result saving
        example['prompt'] = final_prompt_text
    
    return prompt_texts

def get_qwen_first_responses(model: Any, processor: Any, prompt_texts: List[str],
                            messages_list: List[List[Dict]], model_kwargs: Dict) -> List[str]:
    """Generate first responses using Qwen model."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("qwen_vl_utils not available. Please install qwen-vl-utils for Qwen model support.")
    
    model_kwargs_copy = model_kwargs.copy()
    
    image_inputs, video_inputs = process_vision_info(messages_list)
    image_inputs = [[image] for image in image_inputs]
    
    if len(prompt_texts) == 1:
        # Self-consistency mode
        k = model_kwargs_copy.pop('num_return_sequences', 1)
        prompts = prompt_texts * k
        final_image_inputs = image_inputs * k if image_inputs else None
    else:
        # Batch mode
        prompts = prompt_texts
        final_image_inputs = image_inputs
    
    # Process inputs
    processor_inputs = processor(
        text=prompts,
        images=final_image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    input_length = processor_inputs.input_ids.shape[1]
    
    # Generate responses
    extra_args = {"return_dict_in_generate": True, "output_scores": True, "use_cache": True}
    merged_args = {**processor_inputs, **model_kwargs_copy, **extra_args}
    
    with torch.no_grad():
        outputs = model.generate(**merged_args)
    
    responses = processor.batch_decode(
        outputs.sequences[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    # Memory cleanup
    del processor_inputs, image_inputs, video_inputs, outputs
    torch.cuda.empty_cache()
    
    return responses

def get_qwen_second_responses(model: Any, processor: Any, prompt_texts: List[str],
                             first_responses: List[str], messages_list: List[List[Dict]],
                             model_kwargs: Dict, answer_phrase: str) -> List[str]:
    """Generate second responses for answer formatting."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("qwen_vl_utils not available. Please install qwen-vl-utils for Qwen model support.")
    
    model_kwargs_copy = model_kwargs.copy()
    model_kwargs_copy.pop('num_return_sequences', None)
    model_kwargs_copy['do_sample'] = False
    
    image_inputs, video_inputs = process_vision_info(messages_list)
    image_inputs = [[image] for image in image_inputs]
    
    # Remove existing answer phrases from first responses
    first_cut_responses = [resp.split(answer_phrase)[0] for resp in first_responses]
    
    if len(prompt_texts) == 1:
        # Self-consistency mode
        second_prompts = [f"{prompt_texts[0]}{cut_resp} {answer_phrase}"
                         for cut_resp in first_cut_responses]
        final_image_inputs = image_inputs * len(second_prompts) if image_inputs else None
    else:
        # Batch mode
        second_prompts = [f"{prompt_texts[i]}{first_cut_responses[i]} {answer_phrase}"
                         for i in range(len(first_cut_responses))]
        final_image_inputs = image_inputs
    
    # Process inputs
    processor_inputs = processor(
        text=second_prompts,
        images=final_image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    input_length = processor_inputs.input_ids.shape[1]
    
    # Generate responses
    extra_args = {"return_dict_in_generate": True, "output_scores": True, "use_cache": True}
    merged_args = {**processor_inputs, **model_kwargs_copy, **extra_args}
    
    with torch.no_grad():
        outputs = model.generate(**merged_args)
    
    second_responses = processor.batch_decode(
        outputs.sequences[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    # Combine first and second responses
    full_responses = [f"{first_cut_responses[i]} {answer_phrase}{second_responses[i]}"
                     for i in range(len(second_responses))]
    
    # Memory cleanup
    del processor_inputs, image_inputs, video_inputs, outputs
    torch.cuda.empty_cache()
    
    return full_responses

def create_qwen_response_generator(model: Any, processor: Any, mode_type: str):
    """
    Create a response generator function for Qwen model.
    
    This function creates a closure that captures the model, processor, and mode
    and returns a function compatible with the shared evaluation logic.
    
    Args:
        model: Loaded Qwen model
        processor: Loaded Qwen processor  
        mode_type: Prompting mode
        
    Returns:
        Function that generates responses for batches
    """
    def generate_responses(batch_examples: List[Dict], batch_size: int, num_sequences: int) -> List[str]:
        """Generate responses for a batch of examples."""
        torch.cuda.empty_cache()
        
        # Prepare prompts and messages for this batch
        batch_prompts = prepare_qwen_prompts(batch_examples, mode_type)
        batch_messages = prepare_qwen_messages(batch_examples)
        
        # Get generation parameters
        model_kwargs = helpers.get_generation_kwargs(num_sequences)
        
        # Generate responses using two-stage process
        first_responses = get_qwen_first_responses(
            model, processor, batch_prompts, batch_messages, model_kwargs
        )
        full_responses = get_qwen_second_responses(
            model, processor, batch_prompts, first_responses,
            batch_messages, model_kwargs, config.EVAL_ANSWER_PHRASE
        )
        
        return full_responses
    
    return generate_responses

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
    
    # Load Qwen model
    try:
        model, processor = load_qwen_model(args.llm)
        logger.info(f"Successfully loaded Qwen model: {args.llm}")
    except Exception as e:
        logger.error(f"Failed to load Qwen model {args.llm}: {e}", exc_info=True)
        sys.exit(1)
    
    # Create response generator
    response_generator = create_qwen_response_generator(model, processor, args.mode)
    
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