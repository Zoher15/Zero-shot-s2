"""
Helper Functions Module for Zero-shot PGT Evaluation

This module contains all utility functions for the zero-shot prompting evaluation
system, including data loading, prompt building, two-stage model inference,
answer extraction, and metrics calculation.

The module is organized into functional sections:
- Data Loading: Dataset loading and validation
- Prompt Building: Model-specific prompt construction
- Two-Stage Evaluation: Stage 1 (reasoning) + Stage 2 (clean answer)
- Answer Extraction: Response processing and prediction extraction
- Metrics: Evaluation metrics and result saving
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import pandas as pd
from PIL import Image

# Import config
import config

# Logger will be passed as parameter to functions that need it

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_dataset(dataset_name: str, logger) -> List[Dict[str, Any]]:
    """
    Load dataset using configuration registry.

    Args:
        dataset_name: Name of dataset to load
        logger: Logger instance for progress tracking

    Returns:
        List of example dictionaries with 'image', 'question', 'answer' keys

    Raises:
        ValueError: If dataset not supported or no data found
        FileNotFoundError: If dataset files don't exist
    """
    dataset_config = config.get_dataset_config(dataset_name)
    examples = _load_csv_dataset(dataset_config)

    if not examples:
        raise ValueError(f"No examples found for dataset {dataset_name}")

    # Shuffle for consistent evaluation (with seed for reproducibility)
    random.seed(0)
    random.shuffle(examples)

    logger.info(f"Loaded {len(examples)} examples from dataset '{dataset_name}'")
    return examples

def _load_csv_dataset(dataset_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load CSV-based dataset using configuration.

    Args:
        dataset_config: Dataset configuration dictionary

    Returns:
        List of example dictionaries
    """
    csv_path = dataset_config['csv_path']
    image_dir = dataset_config['image_dir']
    image_col = dataset_config['image_col']
    label_col = dataset_config['label_col']

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    examples = []

    for _, row in data.iterrows():
        # Build image path
        if image_dir is not None:
            image_path = str(image_dir / row[image_col])
        else:
            image_path = row[image_col]  # D3 uses full paths

        # Extract label and map to standard format
        raw_label = str(row[label_col]).lower()
        if raw_label == 'real':
            answer = 'real'
        else:
            answer = 'ai-generated'

        examples.append({
            'image': image_path,
            'question': config.BASE_QUESTION,
            'answer': answer
        })

    return examples

# =============================================================================
# PROMPT BUILDING FUNCTIONS
# =============================================================================

def build_prompts_for_examples(examples: List[Dict[str, Any]], phrase_name: str,
                              model_name: str, logger, mode: str = 'prefill') -> List[Dict[str, Any]]:
    """
    Build prompts for all examples.

    Args:
        examples: List of example dictionaries
        phrase_name: Name of phrase to apply
        model_name: Name of model (determines prompt format)
        logger: Logger instance for progress tracking
        mode: Phrase mode - 'prefill', 'prefill-pseudo-system', 'prefill-pseudo-user', 'prompt', or 'instruct' (default: 'prefill')

    Returns:
        List of examples with prompts added
    """
    phrase_text = config.get_phrase_text(phrase_name)

    processed_examples = []
    for example in examples:
        processed_example = _build_single_prompt(example, phrase_text, model_name, mode)
        processed_examples.append(processed_example)

    logger.info(f"Built prompts for {len(processed_examples)} examples using '{mode}' mode with '{phrase_name}'")
    return processed_examples

def _build_single_prompt(example: Dict[str, Any], phrase_text: str, model_name: str,
                        mode: str = 'prefill') -> Dict[str, Any]:
    """
    Build prompt for single example based on mode.

    Args:
        example: Example dictionary
        phrase_text: Phrase text to apply
        model_name: Model name (determines format)
        mode: Phrase mode - 'prefill', 'prefill-pseudo-system', 'prefill-pseudo-user', 'prompt', or 'instruct'

    Returns:
        Example dictionary with messages and metadata added
    """
    model_config = config.get_model_config(model_name)
    messages = []

    # Determine user message text and system message based on mode
    if mode == 'prompt':
        # Append instruction version to question
        instruction = config.get_phrase_instruction_version(phrase_text)
        user_text = example['question'] + (" " + instruction if instruction else "")
    elif mode == 'prefill-pseudo-user':
        # Append "start with" instruction to question
        user_text = example['question']
        if phrase_text:
            user_text += f' Please start your response with "{phrase_text}"'
    elif mode == 'instruct':
        # Add system message with instruction
        instruction = config.get_phrase_instruction_version(phrase_text)
        if instruction:
            messages.append({"role": "system", "content": instruction})
        user_text = example['question']
    elif mode == 'prefill-pseudo-system':
        # Add system message with "start with" instruction
        if phrase_text:
            messages.append({"role": "system", "content": f'Please start your response with "{phrase_text}"'})
        user_text = example['question']
    else:  # prefill
        user_text = example['question']

    # Build user content with image (common logic for all modes)
    # Check if this is a vLLM model or batch API model
    if 'hf_path' in model_config:
        # vLLM models - use model-specific format
        if model_config['hf_path'].startswith('Qwen'):
            content = [
                {"type": "image", "image": example['image']},
                {"type": "text", "text": user_text}
            ]
        elif model_config['hf_path'].startswith('llava-hf'):
            content = [
                {"type": "image"},
                {"type": "text", "text": user_text}
            ]
        else:
            raise ValueError(f"Unknown model format: {model_config['hf_path']}")
    else:
        # Batch API models (Claude) - use generic format
        # Image will be encoded to base64 later in build_claude_batch_requests()
        content = [
            {"type": "image"},
            {"type": "text", "text": user_text}
        ]

    messages.append({"role": "user", "content": content})

    # Only prefill mode needs phrase_text stored for later appending
    return {
        **example,
        'messages': messages,
        'phrase_text': phrase_text if mode == 'prefill' else None,
        'mode': mode
    }

def apply_chat_template_and_phrase(examples: List[Dict[str, Any]], model_name: str, logger) -> List[Dict[str, Any]]:
    """
    Apply chat template using AutoProcessor for all models.

    Args:
        examples: List of examples with messages and phrase_text
        model_name: Model name
        logger: Logger instance for progress tracking

    Returns:
        List of examples with prompts filled
    """
    model_config = config.get_model_config(model_name)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_config['hf_path'],
        trust_remote_code=model_config.get('trust_remote_code', False)
    )

    for example in examples:
        messages = example['messages']

        # Apply chat template - may return string or token IDs depending on model
        prompt_result = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Add phrase text after template (for prefill mode only)
        if example.get('phrase_text'):
            if model_config.get('prefill_mode') == 'append':
                prompt_result += example['phrase_text']
            else:
                raise ValueError(f"Unknown prefill mode: {model_config.get('prefill_mode')}")

        example['prompt'] = prompt_result

    logger.info(f"Applied chat template for {len(examples)} examples")
    return examples

# =============================================================================
# TWO-STAGE EVALUATION FUNCTIONS
# =============================================================================

def evaluate_two_stage(examples: List[Dict[str, Any]], model_name: str, logger, n: int = 1) -> List[Dict[str, Any]]:
    """
    Perform two-stage evaluation: reasoning generation + clean answer extraction.

    Args:
        examples: List of examples with prompts
        model_name: Model name
        logger: Logger instance for progress tracking
        n: Number of responses to generate per input (default: 1)

    Returns:
        List of evaluation results with reasoning traces
    """
    logger.info("🧠 Starting two-stage evaluation...")

    # Initialize LLM engine once for both stages
    model_config = config.get_model_config(model_name)
    from vllm import LLM

    # Build kwargs dynamically
    llm_kwargs = {
        'model': model_config['hf_path'],
        'tensor_parallel_size': model_config['tensor_parallel_size'],
        'gpu_memory_utilization': model_config['gpu_memory_utilization'],
        'trust_remote_code': model_config['trust_remote_code'],
        'max_model_len': model_config.get('max_model_len', 16384),
        'max_num_seqs': model_config.get('max_num_seqs', 256),
        'seed': 0
    }

    # Add optional parameters
    if 'max_num_seqs' in model_config:
        llm_kwargs['max_num_seqs'] = model_config['max_num_seqs']
        logger.info(f"Setting max_num_seqs={model_config['max_num_seqs']}")

    logger.info(f"LLM kwargs: {llm_kwargs}")
    llm_engine = LLM(**llm_kwargs)
    logger.info(f"🚀 Initialized LLM engine for {model_name}")

    # Stage 1: Generate n reasoning responses per example
    logger.info(f"📝 Stage 1: Generating {n} reasoning response(s) per example...")
    stage1_responses_multi = _generate_stage1_responses(examples, model_name, llm_engine, logger, n=n)

    # Stage 2: Extract clean answers for all n responses
    logger.info(f"🎯 Stage 2: Extracting clean answers (temperature=0.0)...")
    results_nested = _generate_stage2_responses_multi(
        examples, stage1_responses_multi, model_name, llm_engine, logger
    )

    # Build final results with nested structure
    results = []
    for example, responses in zip(examples, results_nested):
        # Aggregate predictions
        aggregation = _aggregate_predictions(responses, example['answer'])

        # Build full prompt for logging
        full_prompt = example.get('prompt', '')

        result = {
            'image': example['image'],
            'question': example['question'],
            'full_prompt': full_prompt,
            'ground_truth': example['answer'],
            'responses': responses,
            'aggregated_prediction': aggregation['aggregated_prediction'],
            'aggregated_score': aggregation['aggregated_score'],
            'vote_distribution': aggregation['vote_distribution']
        }
        results.append(result)

    logger.info(f"✅ Completed two-stage evaluation for {len(results)} examples")
    return results

def _generate_stage1_responses(examples: List[Dict[str, Any]], model_name: str, llm_engine, logger, n: int = 1) -> List[List[str]]:
    """
    Generate Stage 1 responses (reasoning).

    Args:
        examples: List of examples with prompts
        model_name: Model name
        llm_engine: Pre-initialized LLM engine
        logger: Logger instance for progress tracking
        n: Number of responses to generate per input (default: 1)

    Returns:
        List of lists: [[resp1, resp2, ...], [resp1, resp2, ...], ...]
        Each inner list has n responses for that example
    """
    model_config = config.get_model_config(model_name)
    from vllm import SamplingParams

    llm = llm_engine
    is_llava = 'llava' in model_config['hf_path'].lower()

    # Prepare inputs
    inputs = []
    valid_indices = []
    for i, example in enumerate(examples):
        try:
            image = Image.open(example['image']).convert('RGB')

            # Skip invalid images for LLaVA only (e.g., 1x1 pixels)
            if is_llava and (image.size[0] < 10 or image.size[1] < 10):
                logger.warning(f"⚠️ Skipping invalid image at index {i}: {example['image']} (size: {image.size})")
                continue

            # Build input item with either prompt (string) or prompt_token_ids (list)
            input_item = {
                "prompt": example['prompt'],
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

    # Build sampling params from model config, falling back to defaults
    max_tokens = model_config.get('stage1_max_tokens', 512)
    model_sampling = model_config.get('sampling_params', {})
    sampling_params = SamplingParams(
        temperature=model_sampling.get('temperature', 0.0 if n == 1 else 1.0),
        top_p=model_sampling.get('top_p', 1.0),
        top_k=model_sampling.get('top_k', 0),
        repetition_penalty=model_sampling.get('repetition_penalty', 1.0),
        presence_penalty=model_sampling.get('presence_penalty', 0.0),
        seed=model_sampling.get('seed', None),
        max_tokens=max_tokens,
        stop=None,
        n=n
    )
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    # Map responses back to original indices for LLaVA, or direct mapping for others
    if is_llava and len(valid_indices) < len(examples):
        responses = [[] for _ in range(len(examples))]  # Create independent empty lists
        for valid_idx, output in zip(valid_indices, outputs):
            responses[valid_idx] = [o.text for o in output.outputs]
        logger.info(f"Generated {len([r for r in responses if r])} examples × {n} Stage 1 responses ({len(examples) - len(valid_indices)} skipped)")
    else:
        responses = [[o.text for o in output.outputs] for output in outputs]
        logger.info(f"Generated {len(responses)} examples × {n} Stage 1 responses")

    return responses

def _generate_stage2_responses_multi(examples: List[Dict[str, Any]], stage1_responses_multi: List[List[str]],
                                     model_name: str, llm_engine, logger) -> List[List[Dict[str, Any]]]:
    """
    Generate Stage 2 responses for all n reasoning traces.

    Args:
        examples: List of examples
        stage1_responses_multi: List of lists [[resp1, resp2, ...], ...]
        model_name: Model name
        llm_engine: Pre-initialized LLM engine
        logger: Logger instance for progress tracking

    Returns:
        List of lists of dicts, each dict has reasoning_response, clean_answer_response, extracted_prediction, score
    """
    # Flatten: create Stage 2 examples for all n×examples
    stage2_examples = []
    example_indices = []
    response_indices = []

    for i, (example, reasoning_list) in enumerate(zip(examples, stage1_responses_multi)):
        # Skip examples with no Stage 1 responses (e.g., invalid images)
        if not reasoning_list:
            continue

        for j, reasoning in enumerate(reasoning_list):
            stage2_example = example.copy()
            stage2_example['prompt'] = example['prompt'] + reasoning + " " + config.ANSWER_PHRASE
            stage2_examples.append(stage2_example)
            example_indices.append(i)
            response_indices.append(j)

    # Generate all Stage 2 responses with temperature=0, n=1
    logger.info(f"Generating {len(stage2_examples)} Stage 2 clean answers (temperature=0.0)...")
    stage2_flat_responses = _generate_stage1_responses(
        stage2_examples, model_name, llm_engine, logger, n=1
    )

    # Unflatten back to [example][response_idx]
    results_nested = [[{} for _ in reasoning_list]
                      for reasoning_list in stage1_responses_multi]

    for idx, clean_answer_list in enumerate(stage2_flat_responses):
        # Handle empty responses from Stage 2 generation failures
        if not clean_answer_list:
            continue

        clean_answer = clean_answer_list[0]  # n=1 for Stage 2
        ex_idx = example_indices[idx]
        resp_idx = response_indices[idx]

        prediction = _find_label_in_text(clean_answer)
        ground_truth = examples[ex_idx]['answer']
        score = 1 if prediction == ground_truth else 0

        results_nested[ex_idx][resp_idx] = {
            'reasoning_response': stage1_responses_multi[ex_idx][resp_idx],
            'clean_answer_response': clean_answer,
            'extracted_prediction': prediction,
            'score': score
        }

    return results_nested

# =============================================================================
# ANSWER EXTRACTION FUNCTIONS
# =============================================================================

def _aggregate_predictions(responses: List[Dict[str, Any]], ground_truth: str) -> Dict[str, Any]:
    """
    Aggregate n responses using majority voting.

    Args:
        responses: List of response dicts with 'extracted_prediction'
        ground_truth: Ground truth label for calculating aggregated score

    Returns:
        {
            'aggregated_prediction': str,
            'aggregated_score': int,
            'vote_distribution': dict
        }
    """
    from collections import Counter

    # Handle skipped/invalid examples with no responses
    if not responses:
        return {
            'aggregated_prediction': '',
            'aggregated_score': 0,
            'vote_distribution': {}
        }

    predictions = [r['extracted_prediction'] for r in responses]
    vote_counts = Counter(predictions)

    # Tie-breaking: prefer 'real'
    if len(vote_counts) > 1:
        most_common = vote_counts.most_common(2)
        if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
            aggregated = 'real'
        else:
            aggregated = most_common[0][0]
    else:
        aggregated = vote_counts.most_common(1)[0][0]

    aggregated_score = 1 if aggregated == ground_truth else 0

    return {
        'aggregated_prediction': aggregated,
        'aggregated_score': aggregated_score,
        'vote_distribution': dict(vote_counts)
    }

def _find_label_in_text(text: str) -> Union[str, None]:
    """
    Find supported label in text.

    Args:
        text: Text to search

    Returns:
        Found label or None
    """
    text_lower = text.lower().strip()

    # Look for exact matches first
    if 'real' in text_lower and 'ai-generated' not in text_lower:
        return 'real'
    elif any(term in text_lower for term in ['ai-generated', 'artificial', 'fake', 'generated']) and 'real' not in text_lower:
        return 'ai-generated'
    elif 'real' in text_lower:
        return 'real'  # Default to real if both present
    return None


# =============================================================================
# METRICS FUNCTIONS
# =============================================================================

def calculate_metrics(results: List[Dict[str, Any]], logger) -> Dict[str, Any]:
    """
    Calculate evaluation metrics from aggregated predictions and individual responses.

    Args:
        results: List of evaluation results with nested responses
        logger: Logger instance for progress tracking

    Returns:
        Dictionary with metrics (aggregated + individual)
    """
    # Extract aggregated predictions
    aggregated_predictions = [r['aggregated_prediction'] for r in results]
    ground_truth = [r['ground_truth'] for r in results]

    # Build confusion matrix for aggregated predictions
    cm = _build_confusion_matrix(aggregated_predictions, ground_truth)

    # Calculate per-class metrics
    class_metrics = _calculate_class_metrics(cm)

    # Calculate macro F1
    macro_f1 = (class_metrics['real']['f1'] + class_metrics['ai-generated']['f1']) / 2

    # Count correct predictions
    correct_count = sum(1 for p, g in zip(aggregated_predictions, ground_truth) if p == g)

    # Aggregated metrics
    metrics = {
        'total_examples': len(results),
        'correct_predictions': correct_count,
        'accuracy': correct_count / len(results),
        'confusion_matrix': cm,
        'class_metrics': class_metrics,
        'macro_f1': macro_f1
    }

    # Calculate individual response metrics (if n>1)
    all_individual_predictions = []
    all_individual_ground_truth = []
    for r in results:
        for resp in r['responses']:
            all_individual_predictions.append(resp['extracted_prediction'])
            all_individual_ground_truth.append(r['ground_truth'])

    if len(all_individual_predictions) > len(results):
        # Only add if n>1
        individual_correct = sum(1 for p, g in zip(all_individual_predictions, all_individual_ground_truth) if p == g)
        metrics['individual_responses'] = {
            'total': len(all_individual_predictions),
            'correct': individual_correct,
            'accuracy': individual_correct / len(all_individual_predictions)
        }
        logger.info(f"Calculated metrics - Aggregated Accuracy: {metrics['accuracy']:.4f}, Individual Accuracy: {metrics['individual_responses']['accuracy']:.4f}, Macro F1: {macro_f1:.4f}")
    else:
        logger.info(f"Calculated metrics - Accuracy: {metrics['accuracy']:.4f}, Macro F1: {macro_f1:.4f}")

    logger.info(f"Confusion matrix: {cm}")

    return metrics

def _build_confusion_matrix(predictions: List[str], ground_truth: List[str]) -> Dict[str, int]:
    """
    Build confusion matrix.

    Args:
        predictions: List of predicted labels
        ground_truth: List of ground truth labels

    Returns:
        Confusion matrix dictionary
    """
    cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    for pred, truth in zip(predictions, ground_truth):
        # Null predictions are treated as incorrect
        if truth == 'ai-generated' and pred == 'ai-generated':
            cm['TP'] += 1
        elif truth == 'real' and pred == 'real':
            cm['TN'] += 1
        elif truth == 'real' and pred != 'real':  # Includes None and 'ai-generated'
            cm['FP'] += 1
        elif truth == 'ai-generated' and pred != 'ai-generated':  # Includes None and 'real'
            cm['FN'] += 1

    return cm

def _calculate_class_metrics(cm: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-class metrics from confusion matrix.

    Args:
        cm: Confusion matrix dictionary

    Returns:
        Dictionary with per-class metrics
    """
    tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']

    # AI-generated class (positive)
    ai_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    ai_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    ai_f1 = 2 * ai_precision * ai_recall / (ai_precision + ai_recall) if (ai_precision + ai_recall) > 0 else 0

    # Real class (negative)
    real_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall) if (real_precision + real_recall) > 0 else 0

    return {
        'ai-generated': {
            'precision': ai_precision,
            'recall': ai_recall,
            'f1': ai_f1
        },
        'real': {
            'precision': real_precision,
            'recall': real_recall,
            'f1': real_f1
        }
    }

def load_reasoning_json(dataset: str, model: str, phrase: str,
                       mode: str = 'prefill', n: int = 1) -> List[Dict[str, Any]]:
    """
    Load reasoning JSON for given configuration.

    Args:
        dataset: Dataset name
        model: Model name
        phrase: Phrase name
        mode: Phrase mode (default: 'prefill')
        n: Number of responses (default: 1)

    Returns:
        List of reasoning results
    """
    output_dir = config.get_output_dir(dataset, model, phrase, mode, n)

    # Find most recent reasoning JSON (exclude _with_probs and _with_logprobs variants)
    reasoning_files = sorted([f for f in output_dir.glob("reasoning_*.json")
                             if '_with_probs' not in f.name and '_with_logprobs' not in f.name])
    if not reasoning_files:
        raise FileNotFoundError(f"No reasoning JSON found in {output_dir}")

    latest_file = reasoning_files[-1]  # Most recent timestamp

    with open(latest_file, 'r') as f:
        data = json.load(f)

    return data

def extract_predictions_and_truth(reasoning_data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Extract aggregated predictions and ground truth from reasoning JSON.

    Args:
        reasoning_data: List of reasoning results

    Returns:
        Tuple of (predictions, ground_truth) lists
    """
    predictions = []
    ground_truth = []

    for result in reasoning_data:
        predictions.append(result['aggregated_prediction'])
        ground_truth.append(result['ground_truth'])

    return predictions, ground_truth

def compute_macro_f1_from_predictions(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Compute macro F1-score from predictions and ground truth.

    Useful for bootstrap CI computation and other statistical analyses.

    Args:
        predictions: List of predicted labels
        ground_truth: List of ground truth labels

    Returns:
        Macro F1-score (average of F1 for each class)
    """
    cm = _build_confusion_matrix(predictions, ground_truth)
    class_metrics = _calculate_class_metrics(cm)
    macro_f1 = (class_metrics['real']['f1'] + class_metrics['ai-generated']['f1']) / 2
    return macro_f1

def save_results(results: List[Dict[str, Any]], metrics: Dict[str, Any],
                output_dir: Path, model_name: str, dataset_name: str,
                phrase_name: str, logger, mode: str = 'prefill', n: int = 1) -> Tuple[Path, Path]:
    """
    Save evaluation results and metrics.

    Args:
        results: List of evaluation results
        metrics: Metrics dictionary
        output_dir: Output directory
        model_name: Model name
        dataset_name: Dataset name
        phrase_name: Phrase name
        logger: Logger instance for progress tracking
        mode: Phrase mode - 'prefill', 'prefill-pseudo-system', 'prefill-pseudo-user', 'prompt', or 'instruct' (default: 'prefill')
        n: Number of responses per input (default: 1)

    Returns:
        Tuple of (reasoning_file_path, performance_file_path)
    """
    from datetime import datetime

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp for consistent file naming with logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save reasoning traces
    reasoning_path = output_dir / f"reasoning_{timestamp}.json"
    with open(reasoning_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Save performance metrics
    performance_data = {
        'model': model_name,
        'dataset': dataset_name,
        'phrase': phrase_name,
        'mode': mode,
        'n_responses': n,
        'metrics': metrics
    }

    performance_path = output_dir / f"performance_fixed_{timestamp}.json"
    with open(performance_path, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2)

    logger.info(f"Reasoning traces saved to: {reasoning_path}")
    logger.info(f"Performance metrics saved to: {performance_path}")

    return reasoning_path, performance_path

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

def create_logger(dataset_name: str, model_name: str, phrase_name: str, mode: str = 'prefill',
                 logger_name: str = 'evaluation', log_suffix: str = '',
                 include_vllm_loggers: bool = True, n: int = 1) -> tuple:
    """
    Create and configure logger with both console and file output.

    Args:
        dataset_name: Name of dataset
        model_name: Name of model
        phrase_name: Name of phrase
        mode: Phrase mode (default: 'prefill')
        logger_name: Logger name (default: 'evaluation')
        log_suffix: Optional suffix for log filename (e.g., '_stage1', '_batch')
        include_vllm_loggers: Whether to configure vLLM loggers (default: True)
        n: Number of responses per input (default: 1)

    Returns:
        Tuple of (configured logger, log file path)
    """
    import logging
    import sys
    from datetime import datetime

    # Create logs directory
    logs_dir = config.get_logs_dir(dataset_name, model_name, phrase_name, mode, n=n)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"evaluation{log_suffix}_{timestamp}.log"

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Configure vLLM loggers to use same handlers (optional)
    if include_vllm_loggers:
        vllm_loggers = [
            'vllm', 'vllm.engine', 'vllm.engine.llm_engine', 'vllm.worker',
            'vllm.model_executor', 'vllm.core', 'vllm.core.scheduler',
            'vllm.distributed', 'vllm.config', 'ray'  # Ray is used by vLLM
        ]
        for vllm_logger_name in vllm_loggers:
            vllm_logger = logging.getLogger(vllm_logger_name)
            vllm_logger.setLevel(logging.INFO)
            vllm_logger.addHandler(console_handler)
            vllm_logger.addHandler(file_handler)

    logger.info(f"📝 Logging to: {log_path}")
    return logger, log_path
