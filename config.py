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
- Dynamic Path Management: Functions for consistent file organization

All paths are defined relative to the project root for portability.
Enhanced with Zero-shot-mod patterns for better organization and consistency.
"""

# Zero-shot-s2/config.py

from pathlib import Path

# Project root directory - anchor for all relative paths
PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# INPUT DATA DIRECTORIES
# =============================================================================

# Base directory for all input datasets
DATA_DIR = PROJECT_ROOT / "data"

# --- D3 Dataset Configuration ---
# Directory containing D3 dataset images and metadata
D3_DIR = DATA_DIR / "d3"
# CSV file containing D3 image IDs for downloading/processing
D3_CSV_FILE = D3_DIR / "2k_sample_ids_d3.csv"  # Ensure filename matches your actual CSV

# --- DF40 Dataset Configuration ---
# Directory containing DF40 dataset with generator subdirectories
DF40_DIR = DATA_DIR / "df40"
# CSV files with image paths and labels for different sample sizes
DF40_10K_CSV_FILE = DF40_DIR / "10k_sample_df40.csv"  # Full 10k sample
DF40_2K_CSV_FILE = DF40_DIR / "2k_sample_df40.csv"    # Reduced 2k sample

# --- GenImage Dataset Configuration ---
# Directory containing GenImage dataset with generator subdirectories
GENIMAGE_DIR = DATA_DIR / "genimage"
# CSV files with image paths and labels for different sample sizes
GENIMAGE_10K_CSV_FILE = GENIMAGE_DIR / "10k_random_sample.csv"  # Full 10k sample
GENIMAGE_2K_CSV_FILE = GENIMAGE_DIR / "2k_random_sample.csv"    # Reduced 2k sample

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

# =============================================================================
# DYNAMIC PATH MANAGEMENT FUNCTIONS
# =============================================================================
# Enhanced with Zero-shot-mod patterns for better organization

def get_model_output_dir(model_name, subdir=None, mode=None, **custom_params):
    """
    Get the output directory for a specific model with dynamic organization.
    
    Enhanced version adopted from Zero-shot-mod for consistent organization.
    
    Args:
        model_name: Name of the model (e.g., 'qwen25-7b', 'llama3-11b')
        subdir: Optional subdirectory ('responses', 'scores', 'plots', 'tables', 'logs')
        mode: Optional mode subdirectory ('zeroshot', 'zeroshot-cot', 'zeroshot-2-artifacts')
        **custom_params: Additional parameters for future extensibility
        
    Returns:
        Path: Path to the model's output directory or subdirectory
        
    Examples:
        get_model_output_dir('qwen25-7b') -> outputs/qwen25-7b/
        get_model_output_dir('qwen25-7b', 'responses') -> outputs/qwen25-7b/responses/
        get_model_output_dir('qwen25-7b', 'responses', 'zeroshot-2-artifacts') -> outputs/qwen25-7b/responses/zeroshot-2-artifacts/
    """
    model_dir = OUTPUTS_DIR / model_name
    if subdir:
        subdir_path = model_dir / subdir
        if mode:
            subdir_path = subdir_path / mode
        return subdir_path
    return model_dir

def get_filename(file_type, dataset, model, mode, num_seq=1, **custom_params):
    """
    Universal filename generator for all evaluation outputs.
    
    Enhanced version adopted from Zero-shot-mod for consistent naming.
    
    Args:
        file_type: Type of file ('responses', 'scores', 'log', etc.)
        dataset: Dataset identifier (e.g., 'genimage2k', 'd32k', 'df402k')
        model: Model name (e.g., 'qwen25-7b', 'llama3-11b')
        mode: Prompting mode (e.g., 'zeroshot', 'zeroshot-cot', 'zeroshot-2-artifacts')
        num_seq: Number of sequences for self-consistency (default: 1)
        **custom_params: Additional parameters for specialized configurations
        
    Returns:
        str: Standardized filename
        
    Examples:
        get_filename('responses', 'genimage2k', 'qwen25-7b', 'zeroshot-2-artifacts', 1)
        -> 'AI_qwen-genimage2k-qwen25-7b-zeroshot-2-artifacts-n1-rationales.jsonl'
    """
    # Build model prefix for consistency with existing naming
    if 'qwen' in model.lower():
        model_prefix = "AI_qwen"
    elif 'llama' in model.lower():
        model_prefix = "AI_llama"
    elif 'code' in model.lower():
        model_prefix = "AI_CoDE"
    else:
        model_prefix = f"AI_{model.split('-')[0] if '-' in model else model}"
    
    # Build base filename
    base = f"{model_prefix}-{dataset}-{model}-{mode}-n{num_seq}"
    
    # Handle custom parameters
    modifiers = []
    DEFAULT_VALUES = {'temperature': 1.0, 'max_tokens': 300, 'top_p': None}
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
    elif file_type == "scores_json":
        return f"{canonical_name}-scores.json"
    elif file_type == "scores_csv":
        return f"{canonical_name}-scores.csv"
    elif file_type == "log":
        return f"{canonical_name}-evaluate.log"
    else:
        return f"{canonical_name}-{file_type}"

# =============================================================================
# MODEL CONFIGURATION SYSTEM
# =============================================================================
# Enhanced model management adopted from Zero-shot-mod patterns

# Supported Vision-Language Models
VLM_MODELS = {
    # Qwen Vision-Language Models
    "qwen25-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen25-7b": "Qwen/Qwen2.5-VL-7B-Instruct", 
    "qwen25-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen25-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    
    # Llama Vision-Language Models
    "llama3-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "llama3-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
}

# Model-specific configurations
MODEL_CONFIGS = {
    'qwen25-3b': {'family': 'qwen', 'size': '3b', 'type': 'vlm'},
    'qwen25-7b': {'family': 'qwen', 'size': '7b', 'type': 'vlm'},
    'qwen25-32b': {'family': 'qwen', 'size': '32b', 'type': 'vlm'},
    'qwen25-72b': {'family': 'qwen', 'size': '72b', 'type': 'vlm'},
    'llama3-11b': {'family': 'llama', 'size': '11b', 'type': 'vlm'},
    'llama3-90b': {'family': 'llama', 'size': '90b', 'type': 'vlm'},
    'code': {'family': 'code', 'size': '6m', 'type': 'cv'},
}

# =============================================================================
# PROMPTING SYSTEM CONFIGURATION
# =============================================================================
# Enhanced prompting system with mode-specific configurations

# Response prefixes for different reasoning modes
RESPONSE_PREFIXES = {
    'zeroshot': '',  # No response prefix
    'zeroshot-cot': '',  # Chain-of-thought handles its own prefix
    'zeroshot-2-artifacts': '',  # Zero-shot-s² handles its own prefix
    'zeroshot-3-artifacts': '',
    'zeroshot-4-artifacts': '',
    'zeroshot-5-artifacts': '',
    'zeroshot-6-artifacts': '',
    'zeroshot-7-artifacts': '',
    'zeroshot-8-artifacts': '',
    'zeroshot-9-artifacts': '',
}

def validate_mode_type(mode_type: str) -> str:
    """
    Validate and normalize mode type.
    
    Args:
        mode_type: Input mode type string
        
    Returns:
        str: Validated and normalized mode type
        
    Raises:
        ValueError: If mode type is not supported
    """
    valid_modes = set(RESPONSE_PREFIXES.keys())
    
    if mode_type not in valid_modes:
        raise ValueError(f"Invalid mode type: {mode_type}. Valid modes: {sorted(valid_modes)}")
    
    return mode_type

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
# Enhanced environment configuration

CONDA_ENV_NAME = "zeroshot_s2"
CONDA_ACTIVATE_PATH = "/data3/zkachwal/miniconda3/etc/profile.d/conda.sh"

# Additional global settings can be added here as needed