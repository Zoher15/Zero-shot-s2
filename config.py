# Zero-shot-s2/config.py

from pathlib import Path

# Project Root - This assumes config.py is in the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Input Data Directories ---
# Base directory for all input data
DATA_DIR = PROJECT_ROOT / "data"

# Specific dataset paths
D3_DIR = DATA_DIR / "D3"
D3_CSV_FILE = D3_DIR / "D3_2k_sample.csv" # Example, make sure filename matches

DF40_DIR = DATA_DIR / "DF40"
DF40_10K_CSV_FILE = DF40_DIR / "10k_sample_df40.csv"
DF40_2K_CSV_FILE = DF40_DIR / "2k_sample_df40.csv"

GENIMAGE_DIR = DATA_DIR / "genimage"
GENIMAGE_10K_CSV_FILE = GENIMAGE_DIR / "10k_random_sample.csv"
GENIMAGE_2K_CSV_FILE = GENIMAGE_DIR / "2k_random_sample.csv"

# --- Output Data Directories ---
# Base directory for all outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RESPONSES_DIR = OUTPUTS_DIR / "responses"
SCORES_DIR = OUTPUTS_DIR / "scores"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TABLES_DIR = OUTPUTS_DIR / "tables"

# --- Cache Directories (if used by scripts for intermediate results) ---
CACHE_DIR = PROJECT_ROOT / "cache" # General cache
F1_PLOT_CACHE_DIR = CACHE_DIR / "f1_cache_plot" # Specific for scaling_consistency.py
CI_CACHE_DIR = CACHE_DIR / "ci_cache" # Specific for macro_f1_bars.py
PROMPT_TABLE_CACHE_DIR = CACHE_DIR / "prompt_table_cache"
PROCESSED_AGGREGATE_DATA_PKL = CACHE_DIR / "processed_model_aggregate_data_v2.pkl"
PROCESSED_INDIVIDUAL_RESPONSES_PKL = CACHE_DIR / "response_cleaned_corpora_v2.pkl"
# Add other cache dirs if needed by other scripts

# --- Log file for load_d3.py ---
# (load_d3.py currently writes 'processing_log.log' to its execution directory.
# We could centralize this too if desired, e.g., PROJECT_ROOT / "processing_log.log"
# or OUTPUTS_DIR / "logs" / "load_d3_processing.log")
LOAD_D3_LOG_FILE = PROJECT_ROOT / "load_d3_processing.log"
EVAL_QUESTION_PHRASE = "Is this image real or AI-generated?"
EVAL_ANSWER_PHRASE = "Final Answer(real/ai-generated):"

# --- Potentially other configurations ---
# Example: NLTK data path if you want to manage it centrally, though NLTK usually handles this.
# NLTK_DATA_DIR = PROJECT_ROOT / "nltk_data"

# You can also add other global settings here if needed later.