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

# Additional global settings can be added here as needed