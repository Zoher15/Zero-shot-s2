"""
Llama Vision-Language Model Evaluation Script

This script evaluates Llama vision-language models for AI-generated image detection
using the Zero-shot-s² framework. It implements a clean functional approach with
shared evaluation logic from helpers.py to eliminate code duplication.

The script uses MllamaForConditionalGeneration from Hugging Face and implements
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
- llama3-11b: Llama 3.2 11B Vision Instruct
- llama3-90b: Llama 3.2 90B Vision Instruct

Usage:
    python experiments/evaluate_AI_llama.py [options]
    
Examples:
    # Basic zero-shot-s² evaluation
    python experiments/evaluate_AI_llama.py -m zeroshot-2-artifacts -d df402k
    
    # Self-consistency evaluation with 5 samples
    python experiments/evaluate_AI_llama.py -m zeroshot-2-artifacts -d df402k -n 5
    
    # Chain-of-thought evaluation
    python experiments/evaluate_AI_llama.py -m zeroshot-cot -d genimage2k -b 10

Command Line Arguments:
    -m, --mode: Prompting mode (default: zeroshot-2-artifacts)
    -llm, --llm: Llama model name (default: llama3-11b)
    -c, --cuda: CUDA device IDs (default: 7)
    -d, --dataset: Dataset to evaluate (default: df402k)
    -b, --batch_size: Batch size for inference (default: 20)
    -n, --num: Number of sequences for self-consistency (default: 1)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import config
from utils import helpers
import argparse

import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import random
from PIL import Image

# =============================================================================
# LLAMA MODEL CONFIGURATION
# =============================================================================

# Supported Llama models
LLAMA_MODELS = {
    "llama3-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "llama3-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
}

# =============================================================================
# LLAMA MODEL FUNCTIONS
# =============================================================================

def load_llama_model(model_name: str) -> Tuple[Any, Any]:
    """
    Load Llama vision-language model and processor.
    
    Args:
        model_name: Name of the Llama model
        
    Returns:
        Tuple of (model, processor)
        
    Raises:
        ValueError: If model name is not supported
    """
    if model_name not in LLAMA_MODELS:
        available_models = list(LLAMA_MODELS.keys())
        raise ValueError(f"Unsupported Llama model: {model_name}. Available: {available_models}")
    
    model_path = LLAMA_MODELS[model_name]
    
    logger.info(f"Loading Llama processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    logger.info(f"Loading Llama model: {model_path}")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    
    logger.info("Compiling Llama model...")
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        logger.info("Llama model compilation complete.")
    except Exception as e:
        logger.warning(f"Llama model compilation failed: {e}. Proceeding without compilation.")
    
    return model, processor

def prepare_llama_prompts(examples: List[Dict], mode_type: str, processor: Any) -> List[str]:
    """
    Prepare prompts for Llama evaluation.
    
    Args:
        examples: List of example dictionaries
        mode_type: Prompting mode
        processor: Llama processor for chat template
        
    Returns:
        List of formatted prompt strings
    """
    instructions_text = helpers.get_instructions_for_mode(mode_type)
    prompt_texts = []
    
    for example in examples:
        messages = []
        if instructions_text:
            messages.append({
                "role": "system", 
                "content": [{"type": "text", "text": instructions_text}]
            })
        messages.append({
            "role": "user", 
            "content": [{"type": "image"}, {"type": "text", "text": example['question']}]
        })
        
        prompt_text = processor.apply_chat_template(
            messages, 
            padding=False, 
            tokenize=False, 
            truncation=True, 
            add_generation_prompt=True
        )
        
        # Add mode-specific prefix
        final_prompt_text = helpers.get_model_guiding_prefix_for_mode(prompt_text, mode_type)
        prompt_texts.append(final_prompt_text)
        
        # Store prompt in example for result saving
        example['prompt'] = final_prompt_text
    
    return prompt_texts

def get_llama_first_responses(model: Any, processor: Any, prompt_texts: List[str], 
                             image_paths: List[str], model_kwargs: Dict) -> List[str]:
    """Generate first responses using Llama model."""
    model_kwargs_copy = model_kwargs.copy()
    
    image_inputs = [[Image.open(path).convert("RGB")] for path in image_paths]
    
    if len(prompt_texts) == 1:
        # Self-consistency mode: replicate prompt for multiple sequences
        k = model_kwargs_copy.pop('num_return_sequences', 1)
        prompts = prompt_texts * k
        final_image_inputs = image_inputs * k
    else:
        # Batch mode: use prompts as-is
        prompts = prompt_texts
        final_image_inputs = image_inputs
    
    # Process inputs
    processor_inputs = processor(
        text=prompts,
        images=final_image_inputs,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False
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
    del processor_inputs, final_image_inputs, outputs
    torch.cuda.empty_cache()
    
    return responses

def get_llama_second_responses(model: Any, processor: Any, prompt_texts: List[str],
                              first_responses: List[str], image_paths: List[str],
                              model_kwargs: Dict, answer_phrase: str) -> List[str]:
    """Generate second responses for answer formatting."""
    model_kwargs_copy = model_kwargs.copy()
    model_kwargs_copy.pop('num_return_sequences', None)
    model_kwargs_copy['do_sample'] = False
    
    image_inputs = [[Image.open(path).convert("RGB")] for path in image_paths]
    
    # Remove existing answer phrases from first responses
    first_cut_responses = [resp.split(answer_phrase)[0] for resp in first_responses]
    
    if len(prompt_texts) == 1:
        # Self-consistency mode
        second_prompts = [f"{prompt_texts[0]}{cut_resp} {answer_phrase}"
                         for cut_resp in first_cut_responses]
        final_image_inputs = image_inputs * len(second_prompts)
    else:
        # Batch mode
        second_prompts = [f"{prompt_texts[i]}{first_cut_responses[i]} {answer_phrase}"
                         for i in range(len(first_cut_responses))]
        final_image_inputs = image_inputs
    
    # Process inputs
    processor_inputs = processor(
        text=second_prompts,
        images=final_image_inputs,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False
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
    del processor_inputs, image_inputs, outputs
    torch.cuda.empty_cache()
    
    return full_responses

def create_llama_response_generator(model: Any, processor: Any, mode_type: str):
    """
    Create a response generator function for Llama model.
    
    This function creates a closure that captures the model, processor, and mode
    and returns a function compatible with the shared evaluation logic.
    
    Args:
        model: Loaded Llama model
        processor: Loaded Llama processor  
        mode_type: Prompting mode
        
    Returns:
        Function that generates responses for batches
    """
    def generate_responses(batch_examples: List[Dict], batch_size: int, num_sequences: int) -> List[str]:
        """Generate responses for a batch of examples."""
        torch.cuda.empty_cache()
        
        # Prepare prompts for this batch
        batch_prompts = prepare_llama_prompts(batch_examples, mode_type, processor)
        batch_image_paths = [ex['image'] for ex in batch_examples]
        
        # Get generation parameters
        model_kwargs = helpers.get_generation_kwargs(num_sequences)
        
        # Generate responses using two-stage process
        first_responses = get_llama_first_responses(
            model, processor, batch_prompts, batch_image_paths, model_kwargs
        )
        full_responses = get_llama_second_responses(
            model, processor, batch_prompts, first_responses, 
            batch_image_paths, model_kwargs, config.EVAL_ANSWER_PHRASE
        )
        
        return full_responses
    
    return generate_responses

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    # Argument parsing
    parser = argparse.ArgumentParser(description="Llama Vision-Language Model Evaluation Script")
    parser.add_argument("-m", "--mode", type=str, default="zeroshot-2-artifacts",
                       help="Prompting mode (default: zeroshot-2-artifacts)")
    parser.add_argument("-llm", "--llm", type=str, default="llama3-11b",
                       help="Llama model name (default: llama3-11b)")
    parser.add_argument("-c", "--cuda", type=str, default="7",
                       help="CUDA device IDs (default: 7)")
    parser.add_argument("-d", "--dataset", type=str, default="df402k",
                       help="Dataset to evaluate (default: df402k)")
    parser.add_argument("-b", "--batch_size", type=int, default=20,
                       help="Batch size for inference (default: 20)")
    parser.add_argument("-n", "--num", type=int, default=1,
                       help="Number of sequences for self-consistency (default: 1)")
    
    args = parser.parse_args()
    
    # Initialize environment
    helpers.initialize_environment(args.cuda)
    
    # Set up logging
    helpers.setup_global_logger(config.EVAL_LLAMA_LOG_FILE)
    global logger
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting Llama evaluation with model: {args.llm}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate model name
    if args.llm not in LLAMA_MODELS:
        available_models = list(LLAMA_MODELS.keys())
        logger.error(f"Unsupported model: {args.llm}. Available models: {available_models}")
        sys.exit(1)
    
    # Load test data
    question_phrase = config.EVAL_QUESTION_PHRASE
    test_data = helpers.load_test_data_unified(args.dataset, question_phrase, config)
    
    if not test_data:
        logger.error(f"Failed to load test data for dataset: {args.dataset}")
        sys.exit(1)
    
    logger.info(f"Successfully loaded {len(test_data)} examples for dataset '{args.dataset}'")
    
    # Load Llama model
    try:
        model, processor = load_llama_model(args.llm)
        logger.info(f"Successfully loaded Llama model: {args.llm}")
    except Exception as e:
        logger.error(f"Failed to load Llama model {args.llm}: {e}", exc_info=True)
        sys.exit(1)
    
    # Create response generator
    response_generator = create_llama_response_generator(model, processor, args.mode)
    
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
            model_prefix="AI_llama",
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