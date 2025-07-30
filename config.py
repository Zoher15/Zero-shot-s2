"""
Central Configuration Module for Zero-shot-s² Repository

This module contains all path configurations, constants, and global parameters
used throughout the Zero-shot-s² project for AI-generated image detection.

The configuration is organized into logical sections:
- Input Data Directories: Paths to datasets (D3, DF40, GenImage)
- Output Data Directories: Paths for experiment results, plots, tables
- Cache Directories: Temporary storage for intermediate processing
- Log Directories: Centralized logging configuration
- Evaluation Constants: Global parameters for evaluation scripts

All paths are defined relative to the project root for portability.
"""

# Zero-shot-s2/config.py

from pathlib import Path
from typing import Dict, Any

# Project root directory - anchor for all relative paths
PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# INPUT DATA DIRECTORIES
# =============================================================================

# Base directory for all input datasets
DATA_DIR = PROJECT_ROOT / "data"

# --- D3 Dataset Configuration ---
# Directory containing D3 dataset images and metadata
# D3_DIR = DATA_DIR / "d3"
D3_DIR = Path("/data3/zkachwal/ELSA_D3")
# CSV file containing D3 image IDs for downloading/processing
# D3 CSV files for different subsets
D3_2K_CSV_FILE = DATA_DIR / "d3" / "d3_2k_sample.csv"  # D3(2k) - 20% subset for additional experiments
D3_7K_CSV_FILE = DATA_DIR / "d3" / "d3_7k_sample.csv"  # D3 main - 80% subset for main evaluation
D3_ALL_CSV_FILE = DATA_DIR / "d3" / "2k_sample_ids_d3.csv"  # Original file with all IDs

# --- DF40 Dataset Configuration ---
# Directory containing DF40 dataset with generator subdirectories
DF40_DIR = Path("/data3/singhdan/DF40")
# CSV files with image paths and labels for different sample sizes
DF40_10K_CSV_FILE = DATA_DIR / "df40" / "10k_sample_df40.csv"  # Full 10k sample
DF40_2K_CSV_FILE = DATA_DIR / "df40" / "2k_sample_df40.csv"    # Reduced 2k sample

# --- GenImage Dataset Configuration ---
# Directory containing GenImage dataset with generator subdirectories
GENIMAGE_DIR = Path("/data3/singhdan/genimage/")
# CSV files with image paths and labels for different sample sizes
GENIMAGE_10K_CSV_FILE = DATA_DIR / "genimage" / "10k_random_sample.csv"  # Full 10k sample
GENIMAGE_2K_CSV_FILE = DATA_DIR / "genimage" / "2k_random_sample.csv"    # Reduced 2k sample

# --- FluxDALL Dataset Configuration ---
# Directory containing FluxDALL dataset with generator subdirectories
FLUXDALL_DIR = Path("/data3/singhdan/AI-GenBench-fake_part/")
# CSV files with image paths and labels for different sample sizes
FLUXDALL_10K_CSV_FILE = DATA_DIR / "fluxdall" / "10k_random_sample.csv"  # Full 10k sample
FLUXDALL_2K_CSV_FILE = DATA_DIR / "fluxdall" / "2k_random_sample.csv"    # Reduced 2k sample

# =============================================================================
# OUTPUT DATA DIRECTORIES
# =============================================================================

# Base directory for all experimental outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --- Experiment Output Subdirectories ---
RESPONSES_DIR = OUTPUTS_DIR / "responses"  # Model responses and rationales (JSONL)
SCORES_DIR = OUTPUTS_DIR / "scores"        # Evaluation metrics (JSON, CSV)
PLOTS_DIR = OUTPUTS_DIR / "plots"          # Generated plots (PNG, PDF)
TABLES_DIR = OUTPUTS_DIR / "tables"        # Generated LaTeX tables (.tex)

# =============================================================================
# CACHE DIRECTORIES
# =============================================================================
# Temporary storage for intermediate processing results to speed up repeated runs

CACHE_DIR = PROJECT_ROOT / "cache"  # General cache directory

# --- Specific Cache Files ---
# Cache for confidence interval calculations in macro_f1_bars.py
CI_CACHE_DIR = CACHE_DIR / "ci_cache"
# Processed aggregate data cache for performance optimization
PROCESSED_AGGREGATE_DATA_PKL = CACHE_DIR / "processed_model_aggregate_data_v2.pkl"
# Processed individual responses cache for text analysis
PROCESSED_INDIVIDUAL_RESPONSES_PKL = CACHE_DIR / "response_cleaned_corpora_v2.pkl"

# =============================================================================
# LOG DIRECTORIES AND FILES
# =============================================================================
# Centralized logging configuration for all scripts

LOGS_DIR = PROJECT_ROOT / "logs"  # Central directory for all log files

# --- Data Processing Script Logs ---
LOAD_D3_LOG_FILE = LOGS_DIR / "load_d3_processing.log"  # D3 dataset downloader

# --- Experiment Script Logs ---
EVAL_QWEN_LOG_FILE = LOGS_DIR / "evaluate_AI_qwen.log"    # Qwen model evaluation
EVAL_LLAMA_LOG_FILE = LOGS_DIR / "evaluate_AI_llama.log"  # Llama model evaluation
EVAL_CODE_LOG_FILE = LOGS_DIR / "evaluate_CoDE.log"       # CoDE model evaluation
EVAL_IMAGES_LOG_FILE = LOGS_DIR / "evaluate_images.log"   # Unified image evaluation

# --- Results Script Logs ---
RESULTS_COMBINE_TABLES_LOG_FILE = LOGS_DIR / "results_combine_tables.log"
RESULTS_DISTINCT_WORDS_LOG_FILE = LOGS_DIR / "results_distinct_words.log"
RESULTS_FIND_IMAGES_LOG_FILE = LOGS_DIR / "results_find_images.log"
RESULTS_MACRO_F1_BARS_LOG_FILE = LOGS_DIR / "results_macro_f1_bars.log"
RESULTS_MODEL_SIZE_TABLE_LOG_FILE = LOGS_DIR / "results_model_size_table.log"
RESULTS_PROMPT_TABLE_LOG_FILE = LOGS_DIR / "results_prompt_table.log"
RESULTS_RECALL_SUBSETS_TABLE_LOG_FILE = LOGS_DIR / "results_recall_subsets_table.log"
RESULTS_SCALING_CONSISTENCY_LOG_FILE = LOGS_DIR / "results_scaling_consistency.log"

# =============================================================================
# EVALUATION CONSTANTS
# =============================================================================
# Global parameters used across evaluation scripts

# Standard question prompt for all evaluations
EVAL_QUESTION_PHRASE = "Is this image real or AI-generated?"

# Response format delimiter for parsing model outputs
EVAL_ANSWER_PHRASE = "Final Answer(real/ai-generated):"

# =============================================================================
# FORMATTING AND DISPLAY OPTIONS
# =============================================================================

# Whether to zero-pad scores in generated tables (e.g., "05.2" vs "5.2")
ZERO_PAD_SCORE_TABLES = True

# Additional global settings can be added here as needed

# =============================================================================
# ENHANCED OUTPUT DIRECTORY MANAGEMENT  
# =============================================================================

def get_model_output_dir(model_name, subdir=None, mode=None, dataset=None, **custom_params):
    """
    Get the output directory for a specific model, optionally with mode organization.
    
    Args:
        model_name: Name of the model (e.g., 'llama3-11b', 'qwen25-7b')
        subdir: Optional subdirectory ('logs', 'responses', 'scores')
        mode: Optional mode subdirectory ('zeroshot', 'zeroshot-cot', 'zeroshot-2-artifacts')
        dataset: Optional dataset subdirectory ('genimage2k', 'df402k', 'd32k')
        **custom_params: Additional parameters for future extensibility
        
    Returns:
        Path: Path to the model's output directory or subdirectory
    """
    model_dir = OUTPUTS_DIR / model_name
    if subdir:
        subdir_path = model_dir / subdir
        if mode:
            subdir_path = subdir_path / mode
        if dataset:
            subdir_path = subdir_path / dataset
        return subdir_path
    return model_dir

def get_filename(file_type, dataset, model, mode, num_seq=1, **custom_params):
    """Universal filename generator for all evaluation outputs."""
    
    # Build base filename
    base = f"{dataset}-{mode}-{model}-n{num_seq}"
    
    # Build modifiers (sorted for consistency)
    modifiers = []
    
    # Handle custom parameters
    DEFAULT_VALUES = {'temperature': 0.0, 'max_tokens': 300, 'top_p': None}
    filtered_params = {k: v for k, v in custom_params.items() 
                      if k not in DEFAULT_VALUES or v != DEFAULT_VALUES[k]}
    
    if filtered_params:
        # Generate semantic abbreviations
        abbrevs = []
        for k, v in sorted(filtered_params.items()):
            if k == 'temperature': abbrevs.append(f"t{str(v).replace('.', '')}")
            elif k == 'max_tokens': abbrevs.append(f"mt{v}")
            elif k == 'top_p': abbrevs.append(f"tp{str(v).replace('.', '')}")
            else: abbrevs.append(f"{k[:2]}{v}")
        modifiers.append('_'.join(abbrevs))
    
    # Assemble canonical name
    canonical_name = f"{base}-{'_'.join(modifiers)}" if modifiers else base
    
    # Add file extensions
    if file_type == "responses":
        return f"{canonical_name}-rationales.jsonl"
    elif file_type == "scores":
        return f"{canonical_name}-scores.json"
    elif file_type == "scores_csv":
        return f"{canonical_name}-scores.csv"
    elif file_type == "log":
        return f"{canonical_name}-evaluate.log"
    else:
        return f"{canonical_name}-{file_type}"

# =============================================================================
# DATASET AND MODE VALIDATION
# =============================================================================

SUPPORTED_DATASETS = [
    "genimage2k", "genimage10k", "df402k", "df4010k", "d32k", "d37k", "fluxdall2k", "fluxdall10k"
]

SUPPORTED_MODES = [
    "zeroshot", "zeroshot-cot", "zeroshot-2-artifacts", "zeroshot-3-artifacts",
    "zeroshot-4-artifacts", "zeroshot-5-artifacts", "zeroshot-6-artifacts",
    "zeroshot-7-artifacts", "zeroshot-8-artifacts", "zeroshot-9-artifacts",
    "sys-cot", "sys-2-artifacts", "ques-cot", "ques-2-artifacts"
]

SUPPORTED_MODELS = [
    "llama3-11b", "llama3-90b", "qwen25-3b", "qwen25-7b", "qwen25-32b", "qwen25-72b", "code", "o3", "gpt-4.1"
]

REASONING_PREFIXES = {
    "zeroshot-cot": "Let's think step by step",
    "zeroshot-visualize": "Let's visualize",
    "zeroshot-examine": "Let's examine",
    "zeroshot-pixel": "Let's examine pixel by pixel",
    "zeroshot-zoom": "Let's zoom in",
    "zeroshot-flaws": "Let's examine the flaws",
    "zeroshot-texture": "Let's examine the textures",
    "zeroshot-style": "Let's examine the style",
    "zeroshot-artifacts": "Let's examine the synthesis artifacts",
    "zeroshot-2-artifacts": "Let's examine the style and the synthesis artifacts",
    "zeroshot-3-artifacts": "Let's examine the synthesis artifacts and the style",
    "zeroshot-4-artifacts": "Let's observe the style and the synthesis artifacts",
    "zeroshot-5-artifacts": "Let's inspect the style and the synthesis artifacts",
    "zeroshot-6-artifacts": "Let's survey the style and the synthesis artifacts",
    "zeroshot-7-artifacts": "Let's scrutinize the style and the synthesis artifacts",
    "zeroshot-8-artifacts": "Let's analyze the style and the synthesis artifacts",
    "zeroshot-9-artifacts": "Let's examine the details and the textures",
}

def validate_dataset_mode(dataset: str, mode: str, model: str = None):
    """
    Validate dataset, mode, and model combinations.
    
    Args:
        dataset: Dataset identifier
        mode: Prompting mode  
        model: Model identifier (optional)
        
    Raises:
        ValueError: If dataset, mode, or model is not supported
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Invalid dataset '{dataset}'. Supported: {SUPPORTED_DATASETS}")
    
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Supported: {SUPPORTED_MODES}")
    
    if model and model not in SUPPORTED_MODELS:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Model '{model}' not in supported list: {SUPPORTED_MODELS}")
    
    return True

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_model_kwargs(model_kwargs):
    """Validate model keyword arguments."""
    if not isinstance(model_kwargs, dict):
        raise ValueError(f"model_kwargs must be a dictionary, got {type(model_kwargs)}")
    
    # Validate numeric parameters
    numeric_params = {
        'max_new_tokens': (1, 8192),
        'temperature': (0.0, 2.0),
        'top_p': (0.0, 1.0),
        'top_k': (1, 1000),
        'repetition_penalty': (0.5, 2.0),
        'num_beams': (1, 50),
        'num_return_sequences': (1, 100)
    }
    
    for param, (min_val, max_val) in numeric_params.items():
        if param in model_kwargs:
            value = model_kwargs[param]
            if value is not None and (value < min_val or value > max_val):
                raise ValueError(f"{param} must be between {min_val} and {max_val}, got {value}")
    
    # Validate boolean parameters
    boolean_params = ['do_sample', 'return_dict_in_generate', 'output_scores']
    for param in boolean_params:
        if param in model_kwargs and not isinstance(model_kwargs[param], bool):
            raise ValueError(f"{param} must be a boolean, got {type(model_kwargs[param])}")
    
    # Validate dependencies
    if model_kwargs.get('do_sample') is True:
        if 'temperature' in model_kwargs and model_kwargs['temperature'] <= 0:
            raise ValueError("temperature must be > 0 when do_sample is True")
    
    if 'num_beams' in model_kwargs and model_kwargs['num_beams'] > 1:
        if model_kwargs.get('do_sample') is True:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Using beam search with sampling may produce inconsistent results")
    
    return True

def get_default_config():
    """Get default configuration with validation."""
    config = {
        'model_kwargs': {
            'max_new_tokens': 300,
            'do_sample': False,
            'temperature': 1.0,
            'top_p': None,
            'top_k': None,
            'repetition_penalty': 1.0
        },
        'mode': 'zeroshot-2-artifacts',
        'batch_size': 20,
        'num_sequences': 1,
        'cuda_devices': '0'
    }
    
    # Validate the default configuration
    validate_model_kwargs(config['model_kwargs'])
    
    return config

def get_generation_kwargs(num_sequences: int) -> Dict[str, Any]:
    """
    Get standard generation parameters for VLM evaluation.
    
    Args:
        num_sequences: Number of sequences for self-consistency
        
    Returns:
        Dictionary of generation parameters
    """
    if num_sequences == 1:
        return {
            "max_new_tokens": 300,
            "do_sample": False,
            "repetition_penalty": 1,
            "top_k": None,
            "top_p": None,
            "temperature": 1
        }
    else:
        return {
            "max_new_tokens": 300,
            "do_sample": True,
            "repetition_penalty": 1,
            "top_k": None,
            "top_p": None,
            "temperature": 1,
            "num_return_sequences": num_sequences
        }