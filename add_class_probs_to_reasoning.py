#!/usr/bin/env python3
"""
Add Class Probabilities to Reasoning JSON (Simplified)

Extends reasoning JSON files with class probabilities at 5 quarterly intervals
(0%, 25%, 50%, 75%, 100%) by generating short completions and tracking token
probabilities.

Approach:
1. Split reasoning into 5 intervals by sentence count
2. For each interval, generate completion after "Final Answer (real/ai-generated): This image is"
3. Extract answer using existing helpers._find_label_in_text() logic
4. Track token probabilities and compute per-interval metrics

Outputs:
- reasoning_{timestamp}_with_conf.json: Per-example class probabilities at each interval
- performance_{timestamp}_with_conf.json: Aggregate metrics (accuracy, F1, confusion matrix) per interval

Usage:
    python add_class_probs_to_reasoning.py --model qwen25-vl-7b --dataset df40 --phrase cot
    python add_class_probs_to_reasoning.py --model qwen3-vl-8b --dataset genimage --phrase cot --mode prompt
"""

import argparse
import json
import os
import sys
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Fix MKL threading layer incompatibility
os.environ.update({
    'MKL_THREADING_LAYER': 'GNU',
    'VLLM_WORKER_MULTIPROC_METHOD': 'fork',
    'VLLM_CONFIGURE_LOGGING': '0'
})

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import helpers
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pysbd

# =============================================================================
# CLI & DATA LOADING
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Add class probabilities to reasoning JSON")
    parser.add_argument('--model', '-m', required=True, choices=config.get_supported_models(),
                       help='VLM model used for evaluation')
    parser.add_argument('--dataset', '-d', required=True, choices=config.get_supported_datasets(),
                       help='Dataset evaluated')
    parser.add_argument('--phrase', '-p', required=True, choices=config.get_supported_phrases(),
                       help='Phrase strategy used')
    parser.add_argument('--mode', default='prefill', choices=config.PHRASE_MODES,
                       help='Phrase mode (default: prefill)')
    parser.add_argument('--cuda', default='0', help='CUDA device IDs (default: 0)')
    parser.add_argument('--debug', action='store_true', help='Process only first 5 examples')
    return parser.parse_args()


# =============================================================================
# SENTENCE SPLITTING
# =============================================================================

def split_reasoning_into_intervals(reasoning_text: str) -> Tuple[List[str], List[int]]:
    """
    Split reasoning text into 5 quarterly interval texts (0%, 25%, 50%, 75%, 100%).

    Args:
        reasoning_text: Full reasoning response text

    Returns:
        Tuple of (interval_texts, interval_sentence_counts)
    """
    segmenter = pysbd.Segmenter(language="en", clean=False)
    sentences = segmenter.segment(reasoning_text)
    num_sentences = len(sentences)

    # Calculate interval indices
    interval_indices = [
        0,
        round(num_sentences * 0.25),
        round(num_sentences * 0.5),
        round(num_sentences * 0.75),
        num_sentences
    ]

    # Build interval texts
    interval_texts = []
    for idx in interval_indices:
        if idx == 0:
            interval_texts.append("")
        else:
            interval_texts.append(" ".join(sentences[:idx]))

    return interval_texts, interval_indices


# =============================================================================
# VLLM INITIALIZATION & INPUT PREPARATION
# =============================================================================

def initialize_llm_and_tokenizer(model_name: str, logger) -> Tuple[LLM, AutoTokenizer]:
    """Initialize LLM engine and tokenizer."""
    logger.info("🚀 Initializing LLM engine for logprob extraction...")

    model_config = config.get_model_config(model_name)

    # Handle vLLM v1 engine flag
    if not model_config.get('use_v1_engine', True):
        os.environ.update({'VLLM_USE_V1': '0', 'MKL_SERVICE_FORCE_INTEL': '1'})
        logger.info("⚙️  Disabling vLLM v1 engine (using legacy v0 for compatibility)")

    llm_kwargs = {
        'model': model_config['hf_path'],
        'tensor_parallel_size': model_config['tensor_parallel_size'],
        'gpu_memory_utilization': model_config['gpu_memory_utilization'],
        'trust_remote_code': model_config['trust_remote_code'],
        'max_model_len': model_config.get('max_model_len', 16384),
        'max_num_seqs': model_config.get('max_num_seqs', 256),
        'seed': 0
    }

    llm_engine = LLM(**llm_kwargs)
    logger.info(f"✅ LLM engine initialized for {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config['hf_path'],
        trust_remote_code=model_config.get('trust_remote_code', False)
    )
    logger.info("✅ Tokenizer loaded")

    return llm_engine, tokenizer


def prepare_vllm_inputs(prompts: List[str], images: List[str], model_name: str, logger) -> Tuple[List[Dict], List[int]]:
    """
    Prepare vLLM inputs from prompts and images.

    Args:
        prompts: List of prompt strings
        images: List of image paths
        model_name: Model name
        logger: Logger instance

    Returns:
        Tuple of (vLLM input dicts, valid_indices)
    """
    model_config = config.get_model_config(model_name)
    is_llava = 'llava' in model_config['hf_path'].lower()

    inputs = []
    valid_indices = []

    for i, (prompt, image_path) in enumerate(zip(prompts, images)):
        try:
            image = Image.open(image_path).convert('RGB')

            # Skip invalid images for LLaVA only
            if is_llava and (image.size[0] < 10 or image.size[1] < 10):
                logger.warning(f"⚠️ Skipping invalid image at index {i}: {image_path} (size: {image.size})")
                continue

            input_item = {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
                "multi_modal_uuids": {"image": f"uuid_{i}"}
            }
            inputs.append(input_item)
            valid_indices.append(i)

        except Exception as e:
            if is_llava:
                logger.warning(f"⚠️ Skipping image at index {i} due to error: {str(e)}")
                continue
            else:
                raise

    if not inputs:
        raise ValueError("No valid images to process")

    return inputs, valid_indices


# =============================================================================
# ANSWER EXTRACTION WITH PROBABILITIES
# =============================================================================

def extract_answer_with_probs(prompts: List[str], images: List[str], model_name: str,
                              llm_engine, tokenizer, logger) -> List[Dict]:
    """
    Generate short completions and extract answers with token probabilities.

    Args:
        prompts: List of prompt strings ending with "This image is"
        images: List of image paths
        model_name: Model name
        llm_engine: Pre-initialized LLM engine
        tokenizer: Tokenizer instance
        logger: Logger instance

    Returns:
        List of dicts with generated_text, token_probs, extracted_answer
        Returns None for skipped examples (LLaVA invalid images)
    """
    model_config = config.get_model_config(model_name)
    is_llava = 'llava' in model_config['hf_path'].lower()

    inputs, valid_indices = prepare_vllm_inputs(prompts, images, model_name, logger)

    # Generate short completion (stop at '.')
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
        stop=['.'],
        logprobs=1,  # Top-1 prob per token
        n=1
    )
    outputs = llm_engine.generate(inputs, sampling_params=sampling_params)

    # Extract tokens, probabilities, and answers
    results = [None] * len(prompts)  # Pre-fill with None for skipped indices

    for valid_idx, output in zip(valid_indices, outputs):
        if output.outputs and output.outputs[0].logprobs:
            generated_text = output.outputs[0].text

            # Extract tokens, token IDs, and probabilities
            tokens = []
            token_ids = []
            token_probs = []
            for logprob_dict in output.outputs[0].logprobs:
                # logprob_dict maps token_id -> Logprob object
                if logprob_dict:
                    top_token_id = list(logprob_dict.keys())[0]
                    logprob_obj = logprob_dict[top_token_id]
                    prob = math.exp(logprob_obj.logprob) if logprob_obj.logprob != -float('inf') else 0.0

                    tokens.append(logprob_obj.decoded_token)
                    token_ids.append(top_token_id)
                    token_probs.append(prob)

            # Use existing extraction logic from helpers
            extracted_answer = helpers._find_label_in_text(generated_text)

            results[valid_idx] = {
                'generated_text': generated_text,
                'tokens': tokens,
                'token_ids': token_ids,
                'token_probs': token_probs,
                'extracted_answer': extracted_answer
            }

    if is_llava:
        logger.info(f"Extracted answers for {len([r for r in results if r is not None])} examples ({len(prompts) - len(valid_indices)} skipped)")
    else:
        logger.info(f"Extracted answers for {len(results)} examples")

    return results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def add_class_probs_to_examples(reasoning_data: List[Dict], model_name: str, mode: str, logger) -> List[Dict]:
    """
    Add class probabilities to reasoning data at 5 quarterly intervals.

    Args:
        reasoning_data: List of reasoning results (from reasoning JSON)
        model_name: Model name
        mode: Phrase mode
        logger: Logger instance

    Returns:
        Extended reasoning data with class_probs_intervals added
    """
    # Initialize LLM engine once
    llm_engine, tokenizer = initialize_llm_and_tokenizer(model_name, logger)

    # Pre-compute interval texts for all examples
    logger.info("✂️  Splitting reasoning responses into quarterly intervals (0%, 25%, 50%, 75%, 100%)...")
    interval_data_by_example = []

    for ex in reasoning_data:
        reasoning_text = ex['responses'][0]['reasoning_response']
        interval_texts, interval_sentence_counts = split_reasoning_into_intervals(reasoning_text)

        interval_data_by_example.append({
            'full_prompt': ex['full_prompt'],
            'image': ex['image'],
            'interval_texts': interval_texts,
            'interval_sentence_counts': interval_sentence_counts,
            'ground_truth': ex['ground_truth']
        })

    # Initialize storage
    for ex in reasoning_data:
        ex['class_probs_intervals'] = []

    # Process each interval separately
    interval_labels = [0.00, 0.25, 0.50, 0.75, 1.00]

    for interval_idx, interval_label in enumerate(interval_labels):
        logger.info(f"📊 Processing interval {interval_label:.2f} ({interval_idx+1}/5)...")

        # Build prompts for this interval across all examples
        interval_prompts = []
        interval_images = []

        for ex_data in interval_data_by_example:
            interval_text = ex_data['interval_texts'][interval_idx]

            if interval_text:
                prompt = ex_data['full_prompt'] + interval_text + f" {config.ANSWER_PHRASE} This image is"
            else:
                # Interval 0.0 (no reasoning yet)
                if mode == 'prefill':
                    prompt = ex_data['full_prompt'] + f". {config.ANSWER_PHRASE} This image is"
                else:
                    prompt = ex_data['full_prompt'] + f" {config.ANSWER_PHRASE} This image is"

            interval_prompts.append(prompt)
            interval_images.append(ex_data['image'])

        # Generate answers for all examples at this interval
        logger.info(f"   Generating answers for {len(interval_prompts)} examples at interval {interval_label:.2f}...")
        results = extract_answer_with_probs(interval_prompts, interval_images, model_name, llm_engine, tokenizer, logger)

        # Store results for this interval
        for ex_idx, result in enumerate(results):
            ground_truth = interval_data_by_example[ex_idx]['ground_truth']
            sentence_count = interval_data_by_example[ex_idx]['interval_sentence_counts'][interval_idx]

            if result is None:
                # Skipped example (LLaVA invalid image)
                interval_result = {
                    'interval': interval_label,
                    'num_sentences': sentence_count,
                    'generated_text': None,
                    'tokens': None,
                    'token_ids': None,
                    'token_probs': None,
                    'extracted_answer': None,
                    'score': 0
                }
            else:
                score = 1 if result['extracted_answer'] == ground_truth else 0

                interval_result = {
                    'interval': interval_label,
                    'num_sentences': sentence_count,
                    'generated_text': result['generated_text'],
                    'tokens': result['tokens'],
                    'token_ids': result['token_ids'],
                    'token_probs': result['token_probs'],
                    'extracted_answer': result['extracted_answer'],
                    'score': score
                }

            reasoning_data[ex_idx]['class_probs_intervals'].append(interval_result)

    logger.info(f"✅ Added class probabilities at 5 intervals for {len(reasoning_data)} examples")
    return reasoning_data


# =============================================================================
# STATISTICS CALCULATION
# =============================================================================

def calculate_interval_statistics(reasoning_data: List[Dict], logger) -> Dict[str, Dict[str, Any]]:
    """
    Calculate aggregate statistics per interval including confusion matrix and metrics.

    Args:
        reasoning_data: List of reasoning results with class_probs_intervals
        logger: Logger instance

    Returns:
        Dict mapping interval labels to statistics
    """
    import numpy as np

    interval_labels = ["0.00", "0.25", "0.50", "0.75", "1.00"]
    interval_statistics = {}

    for interval_idx, interval_label in enumerate(interval_labels):
        # Extract predictions and ground truth for this interval
        predictions = []
        ground_truth_labels = []
        first_token_probs = []
        avg_token_probs = []

        for ex in reasoning_data:
            if 'class_probs_intervals' in ex and len(ex['class_probs_intervals']) > interval_idx:
                interval_data = ex['class_probs_intervals'][interval_idx]

                # Skip None results (LLaVA skipped images)
                if interval_data['extracted_answer'] is not None:
                    predictions.append(interval_data['extracted_answer'])
                    ground_truth_labels.append(ex['ground_truth'])

                    # Extract probability statistics
                    if interval_data['token_probs']:
                        first_token_probs.append(interval_data['token_probs'][0])
                        avg_token_probs.append(np.mean(interval_data['token_probs']))

        # Build confusion matrix using helpers
        cm = helpers._build_confusion_matrix(predictions, ground_truth_labels)
        class_metrics = helpers._calculate_class_metrics(cm)

        # Calculate accuracy and macro F1
        total = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
        accuracy = (cm['TP'] + cm['TN']) / total if total > 0 else 0.0
        macro_f1 = (class_metrics['real']['f1'] + class_metrics['ai-generated']['f1']) / 2

        # Compute probability statistics
        def compute_stats(values):
            if not values:
                return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
            arr = np.array(values)
            return {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'count': len(values)
            }

        first_token_stats = compute_stats(first_token_probs)
        avg_token_stats = compute_stats(avg_token_probs)

        interval_statistics[interval_label] = {
            'total_examples': len(predictions),
            'correct_predictions': cm['TP'] + cm['TN'],
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'confusion_matrix': cm,
            'class_metrics': class_metrics,
            'first_token_prob_stats': first_token_stats,
            'avg_token_prob_stats': avg_token_stats
        }

        logger.info(f"📊 Interval {interval_label} - Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}, "
                   f"First token prob: {first_token_stats['mean']:.4f} ± {first_token_stats['std']:.4f}")

    return interval_statistics


# =============================================================================
# FILE SAVING
# =============================================================================

def save_extended_files(reasoning_data: List[Dict], interval_statistics: Dict[str, Dict[str, Any]],
                       dataset_name: str, model_name: str, phrase_name: str, mode: str,
                       logger) -> Tuple[Path, Path]:
    """
    Save extended reasoning JSON and performance JSON.

    Args:
        reasoning_data: Extended reasoning data with class_probs_intervals
        interval_statistics: Dict mapping interval labels to statistics
        dataset_name: Dataset name
        model_name: Model name
        phrase_name: Phrase name
        mode: Phrase mode
        logger: Logger instance

    Returns:
        Tuple of (reasoning_path, performance_path)
    """
    output_dir = config.get_output_dir(dataset_name, model_name, phrase_name, mode, n=1)

    # Find original reasoning file to extract timestamp
    reasoning_files = sorted([f for f in output_dir.glob("reasoning_*.json")
                             if '_with_probs' not in f.name])
    if reasoning_files:
        original_file = reasoning_files[-1]
        timestamp = original_file.stem.split('_', 1)[1]  # Extract 'YYYYMMDD_HHMMSS'
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save extended reasoning JSON
    reasoning_path = output_dir / f"reasoning_{timestamp}_with_conf.json"
    with open(reasoning_path, 'w', encoding='utf-8') as f:
        json.dump(reasoning_data, f, indent=2)
    logger.info(f"💾 Saved extended reasoning JSON: {reasoning_path}")

    # Save performance JSON with interval statistics
    performance_data = {
        'model': model_name,
        'dataset': dataset_name,
        'phrase': phrase_name,
        'mode': mode,
        'n_responses': 1,
        'intervals': interval_statistics
    }

    performance_path = output_dir / f"performance_{timestamp}_with_conf.json"
    with open(performance_path, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2)
    logger.info(f"💾 Saved performance with interval statistics: {performance_path}")

    return reasoning_path, performance_path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main pipeline for adding class probabilities to reasoning JSON."""
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    logger, log_path = helpers.create_logger(
        args.dataset, args.model, args.phrase, args.mode,
        logger_name='class_probs', log_suffix='_probs',
        include_vllm_loggers=True, n=1
    )

    logger.info("🚀 Starting Class Probability Extraction")
    logger.info(f"Model: {args.model} | Dataset: {args.dataset} | Phrase: {args.phrase} (mode: {args.mode})")
    logger.info(f"CUDA devices: {args.cuda}")

    try:
        # Load reasoning JSON
        logger.info("📂 Loading reasoning JSON...")
        reasoning_data = helpers.load_reasoning_json(
            args.dataset, args.model, args.phrase, args.mode, n=1
        )

        # Validate reasoning data structure
        logger.info("🔍 Validating reasoning data structure...")
        original_count = len(reasoning_data)
        valid_reasoning_data = []
        for i, ex in enumerate(reasoning_data):
            if 'responses' not in ex or not ex['responses']:
                logger.warning(f"⚠️ Skipping example {i} with missing/empty responses")
                continue
            if not ex['responses'][0].get('reasoning_response'):
                logger.warning(f"⚠️ Skipping example {i} with empty reasoning_response")
                continue
            valid_reasoning_data.append(ex)

        reasoning_data = valid_reasoning_data
        skipped_count = original_count - len(reasoning_data)
        if skipped_count > 0:
            logger.info(f"⚠️ Skipped {skipped_count} malformed examples (kept {len(reasoning_data)}/{original_count})")
        else:
            logger.info(f"✅ All {len(reasoning_data)} examples have valid structure")

        if args.debug:
            reasoning_data = reasoning_data[:5]
            logger.info(f"🐛 Debug mode: Processing only {len(reasoning_data)} examples")

        # Extract and add class probabilities
        logger.info("🧠 Extracting class probabilities at 5 intervals...")
        extended_data = add_class_probs_to_examples(
            reasoning_data, args.model, args.mode, logger
        )

        # Calculate interval statistics
        logger.info("📊 Calculating statistics for each interval...")
        interval_statistics = calculate_interval_statistics(extended_data, logger)

        # Save extended files
        logger.info("💾 Saving extended files...")
        reasoning_path, performance_path = save_extended_files(
            extended_data, interval_statistics,
            args.dataset, args.model, args.phrase, args.mode, logger
        )

        # Display summary
        logger.info("\n" + "="*70)
        logger.info("✅ CLASS PROBABILITY EXTRACTION COMPLETED")
        logger.info("="*70)
        logger.info(f"Extended reasoning: {reasoning_path}")
        logger.info(f"Performance (interval stats): {performance_path}")
        logger.info(f"Examples processed: {len(extended_data)}")
        logger.info(f"Intervals analyzed: 0.00, 0.25, 0.50, 0.75, 1.00")
        # Show stats for final interval
        if '1.00' in interval_statistics:
            final_stats = interval_statistics['1.00']
            logger.info(f"Final interval (1.00) accuracy: {final_stats['accuracy']:.4f}")
            logger.info(f"Final interval (1.00) macro F1: {final_stats['macro_f1']:.4f}")
            logger.info(f"Final interval (1.00) avg first token prob: {final_stats['first_token_prob_stats']['mean']:.4f}")
        logger.info(f"Log file: {log_path}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"💥 Extraction failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
