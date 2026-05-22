"""
Configuration Module for Zero-shot PGT Evaluation

This module contains all configuration registries, constants, and configuration
management functions for the zero-shot prompting evaluation system.

Features:
- VLM model configurations (Qwen2.5-VL, LLaVA-OneVision, Qwen3-VL)
- Dataset configurations with paths and metadata
- Phrase configurations for different prompting strategies
- Validation and getter functions
"""

from pathlib import Path
from typing import Dict, Any

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# DATASET IMAGE DIRECTORIES
# =============================================================================

DF40_IMAGE_DIR = Path("/data/df40/")
GENIMAGE_IMAGE_DIR = Path("/data/genimage/")
D3_IMAGE_DIR = DATA_DIR / "d3" / "images"  # D3 images downloaded from HuggingFace
# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

VLM_MODELS = {
    'qwen25-vl-7b': {
        'hf_path': "Qwen/Qwen2.5-VL-7B-Instruct",
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': 0.92,
        'trust_remote_code': True,
        'prefill_mode': 'append',  # Append prefill after chat template
        'type': 'vllm'  # Local vLLM inference
    },
    'llava-onevision-7b': {
        'hf_path': "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': 0.92,
        'trust_remote_code': True,
        'prefill_mode': 'append',  # Append prefill after chat template
        'type': 'vllm'  # Local vLLM inference
    },
    'qwen3-vl-8b': {
        'hf_path': "Qwen/Qwen3-VL-8B-Instruct",
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': 0.92,
        'trust_remote_code': True,
        'prefill_mode': 'append',
        'type': 'vllm',
        'stage1_max_tokens': 1024,  # Instruct model is chattier than Qwen2.5
        # 'sampling_params': {  # From Qwen3-VL-8B-Instruct model card (presence_penalty removed)
        #     'temperature': 0.7,
        #     'top_p': 0.8,
        #     'top_k': 20,
        #     'repetition_penalty': 1.0,
        #     'seed': 0
        # }
    },
    'qwen3-vl-8b-thinking': {
        'hf_path': "Qwen/Qwen3-VL-8B-Thinking",
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': 0.92,
        'trust_remote_code': True,
        'prefill_mode': 'append',
        'type': 'vllm',
        'stage1_max_tokens': 4096,  # Thinking model needs more tokens for <think> block
        'sampling_params': {  # From Qwen3-VL-8B-Thinking model card
            'temperature': 1.0,
            'top_p': 0.95,
            'top_k': 20,
            'repetition_penalty': 1.0,
            'presence_penalty': 0.0,
            'seed': 0
        }
    }
}

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASETS = {
    'df40': {
        'csv_path': DATA_DIR / "df40" / "10k_sample_df40.csv",
        'image_dir': DF40_IMAGE_DIR,
        'image_col': 'img_path',
        'label_col': 'dataset',
        'answer_mapping': {'real': 'real', 'ai-generated': 'ai-generated'}
    },
    'df40-2k': {
        'csv_path': DATA_DIR / "df40" / "2k_sample_df40.csv",
        'image_dir': DF40_IMAGE_DIR,
        'image_col': 'img_path',
        'label_col': 'dataset',
        'answer_mapping': {'real': 'real', 'ai-generated': 'ai-generated'}
    },
    'd3': {
        'csv_path': DATA_DIR / "d3" / "7k_sample.csv",
        'image_dir': None,  # D3 uses full paths in CSV
        'image_col': 'image',
        'label_col': 'answer',
        'answer_mapping': {'real': 'real', 'ai-generated': 'ai-generated'}
    },
    'd3-2k': {
        'csv_path': DATA_DIR / "d3" / "2k_sample.csv",
        'image_dir': None,  # D3 uses full paths in CSV
        'image_col': 'image',
        'label_col': 'answer',
        'answer_mapping': {'real': 'real', 'ai-generated': 'ai-generated'}
    },
    'genimage': {
        'csv_path': DATA_DIR / "genimage" / "10k_random_sample.csv",
        'image_dir': GENIMAGE_IMAGE_DIR,
        'image_col': 'img_path',
        'label_col': 'dataset',
        'answer_mapping': {'real': 'real', 'ai-generated': 'ai-generated'}
    },
    'genimage-2k': {
        'csv_path': DATA_DIR / "genimage" / "2k_random_sample.csv",
        'image_dir': GENIMAGE_IMAGE_DIR,
        'image_col': 'img_path',
        'label_col': 'dataset',
        'answer_mapping': {'real': 'real', 'ai-generated': 'ai-generated'}
    }
}

# =============================================================================
# PHRASE CONFIGURATIONS
# =============================================================================

PHRASES = {
    'baseline': "",  # No prefill
    'cot': "Let's think step by step",
    'artifacts': "Let's examine the synthesis artifacts",
    'style': "Let's examine the style",
    'o2': "Let's observe the style and the synthesis artifacts",
    's2': "Let's examine the style and the synthesis artifacts",
    's3': "Let's scrutinize the style and the synthesis artifacts",
    's4': "Let's analyze the style and the synthesis artifacts",
    's5': "Let's inspect the style and the synthesis artifacts",
    's6': "Let's survey the style and the synthesis artifacts",
    's7': "Let's examine the synthesis artifacts and the style",
    'details': "Let's examine the details",
    'flaws': "Let's examine the flaws",
    'pixel': "Let's examine pixel by pixel",
    'zoom': "Let's zoom in",
    'texture': "Let's examine the textures"
}

# =============================================================================
# EVALUATION CONSTANTS
# =============================================================================

BASE_QUESTION = "Is this image real or AI-generated?"
ANSWER_PHRASE = "Final Answer (real/ai-generated):"
SUPPORTED_LABELS = ['real', 'ai-generated']

# Phrase mode options
PHRASE_MODES = ['prefill', 'prefill-pseudo-system', 'prefill-pseudo-user', 'prompt', 'instruct']

# =============================================================================
# CONFIGURATION GETTER FUNCTIONS
# =============================================================================

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get model configuration by name.

    Args:
        model_name: Name of the model (e.g., 'qwen25-vl-7b', 'qwen3-vl-8b')

    Returns:
        Dictionary containing model configuration

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in VLM_MODELS:
        supported_models = list(VLM_MODELS.keys())
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported_models}")

    return VLM_MODELS[model_name]

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset configuration by name.

    Args:
        dataset_name: Name of the dataset (e.g., 'df40', 'd3', 'genimage')

    Returns:
        Dictionary containing dataset configuration

    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in DATASETS:
        supported_datasets = list(DATASETS.keys())
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported_datasets}")

    return DATASETS[dataset_name]

def get_phrase_text(phrase_name: str) -> str:
    """
    Get phrase text by name.

    Args:
        phrase_name: Name of the phrase (e.g., 'cot', 'artifacts')

    Returns:
        Phrase text string

    Raises:
        ValueError: If phrase_name is not supported
    """
    if phrase_name not in PHRASES:
        supported_phrases = list(PHRASES.keys())
        raise ValueError(f"Unsupported phrase '{phrase_name}'. Supported: {supported_phrases}")

    return PHRASES[phrase_name]

def get_phrase_instruction_version(phrase_text: str) -> str:
    """
    Convert phrase text to instruction version.

    Args:
        phrase_text: Original phrase text (e.g., "Let's think step by step")

    Returns:
        Instruction version (e.g., "Please think step by step.")
    """
    if not phrase_text:
        return ""

    # Replace "Let's" with "Please" and add period
    instruction_version = phrase_text.replace("Let's", "Please")

    if not instruction_version.endswith('.'):
        instruction_version += '.'

    return instruction_version

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config_combination(model_name: str, dataset_name: str, phrase_name: str) -> bool:
    """
    Validate that model, dataset, and phrase combination is supported.

    Args:
        model_name: Model name to validate
        dataset_name: Dataset name to validate
        phrase_name: Phrase name to validate

    Returns:
        True if combination is valid

    Raises:
        ValueError: If any component is not supported
    """
    # This will raise ValueError if any component is invalid
    get_model_config(model_name)
    get_dataset_config(dataset_name)
    get_phrase_text(phrase_name)

    return True

def get_supported_models() -> list:
    """Get all supported models."""
    return list(VLM_MODELS.keys())

def get_supported_datasets() -> list:
    """Get all supported datasets."""
    return list(DATASETS.keys())

def get_supported_phrases() -> list:
    """Get all supported phrases."""
    return list(PHRASES.keys())

# =============================================================================
# OUTPUT PATH FUNCTIONS
# =============================================================================


def get_output_dir(dataset_name: str, model_name: str, phrase_name: str, mode: str = 'prefill', n: int = 1) -> Path:
    """
    Get output directory with hierarchical structure: output/dataset/model/phrase/mode/n={n}/

    Args:
        dataset_name: Name of dataset
        model_name: Name of model
        phrase_name: Name of phrase strategy
        mode: Phrase mode - 'prefill', 'prefill-pseudo-system', 'prefill-pseudo-user', 'prompt', or 'instruct' (default: 'prefill')
        n: Number of responses per input (default: 1)
    """
    if mode not in PHRASE_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Supported: {PHRASE_MODES}")

    # Baseline has no mode subdirectory (all modes are equivalent for empty phrase)
    if phrase_name == 'baseline':
        if n == 1:
            output_dir = OUTPUT_DIR / dataset_name / model_name / phrase_name
        else:
            output_dir = OUTPUT_DIR / dataset_name / model_name / phrase_name / f"n={n}"
    else:
        if n == 1:
            output_dir = OUTPUT_DIR / dataset_name / model_name / phrase_name / mode
        else:
            output_dir = OUTPUT_DIR / dataset_name / model_name / phrase_name / mode / f"n={n}"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_logs_dir(dataset_name: str, model_name: str, phrase_name: str, mode: str = 'prefill', n: int = 1) -> Path:
    """
    Get logs directory with hierarchical structure: logs/dataset/model/phrase/mode/n={n}/

    Args:
        dataset_name: Name of dataset
        model_name: Name of model
        phrase_name: Name of phrase strategy
        mode: Phrase mode - 'prefill', 'prefill-pseudo-system', 'prefill-pseudo-user', 'prompt', or 'instruct' (default: 'prefill')
        n: Number of responses per input (default: 1)
    """
    if mode not in PHRASE_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Supported: {PHRASE_MODES}")

    # Baseline has no mode subdirectory (all modes are equivalent for empty phrase)
    if phrase_name == 'baseline':
        if n == 1:
            logs_dir = LOGS_DIR / dataset_name / model_name / phrase_name
        else:
            logs_dir = LOGS_DIR / dataset_name / model_name / phrase_name / f"n={n}"
    else:
        if n == 1:
            logs_dir = LOGS_DIR / dataset_name / model_name / phrase_name / mode
        else:
            logs_dir = LOGS_DIR / dataset_name / model_name / phrase_name / mode / f"n={n}"

    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir
