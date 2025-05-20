# Zero-shot-s2/config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Input Data Directories ---
# Base directory for all input data
DATA_DIR = PROJECT_ROOT / "data"

# Specific dataset paths
# # Specific dataset paths
# D3_DIR = DATA_DIR / "d3"
# D3_CSV_FILE = D3_DIR / "2k_sample_ids_d3.csv" # Example, make sure filename matches

# DF40_DIR = DATA_DIR / "df40"
# DF40_10K_CSV_FILE = DF40_DIR / "10k_sample_df40.csv"
# DF40_2K_CSV_FILE = DF40_DIR / "2k_sample_df40.csv"

# GENIMAGE_DIR = DATA_DIR / "genimage"
# GENIMAGE_10K_CSV_FILE = GENIMAGE_DIR / "10k_random_sample.csv"
# GENIMAGE_2K_CSV_FILE = GENIMAGE_DIR / "2k_random_sample.csv"

D3_DIR = "/data3/zkachwal/ELSA_D3/"
D3_CSV_FILE = "/data3/zkachwal/Zero-shot-s2/data/d3/2k_sample_ids_d3.csv" # Example, make sure filename matches

DF40_DIR = "/data3/singhdan/DF40/"
DF40_10K_CSV_FILE = "/data3/zkachwal/Zero-shot-s2/data/df40/10k_sample_df40.csv"
DF40_2K_CSV_FILE = "/data3/zkachwal/Zero-shot-s2/data/df40/2k_sample_df40.csv"

GENIMAGE_DIR = "/data3/singhdan/genimage/"
GENIMAGE_10K_CSV_FILE = "/data3/zkachwal/Zero-shot-s2/data/genimage/10k_random_sample.csv"
GENIMAGE_2K_CSV_FILE = "/data3/zkachwal/Zero-shot-s2/data/genimage/2k_random_sample.csv"

# --- Output Data Directories ---
# Base directory for all outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RESPONSES_DIR = OUTPUTS_DIR / "responses"
SCORES_DIR = OUTPUTS_DIR / "scores"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TABLES_DIR = OUTPUTS_DIR / "tables"

# --- Cache Directories (if used by scripts for intermediate results) ---
CACHE_DIR = PROJECT_ROOT / "cache" # General cache
# F1_PLOT_CACHE_DIR = CACHE_DIR / "f1_cache_plot" # Removed as scaling_consistency.py will no longer use it
CI_CACHE_DIR = CACHE_DIR / "ci_cache" # Specific for macro_f1_bars.py
PROCESSED_AGGREGATE_DATA_PKL = CACHE_DIR / "processed_model_aggregate_data_v2.pkl"
PROCESSED_INDIVIDUAL_RESPONSES_PKL = CACHE_DIR / "response_cleaned_corpora_v2.pkl"
# Add other cache dirs if needed by other scripts

# --- Log file for load_d3.py ---
LOAD_D3_LOG_FILE = PROJECT_ROOT / "load_d3_processing.log"
EVAL_QUESTION_PHRASE = "Is this image real or AI-generated?"
EVAL_ANSWER_PHRASE = "Final Answer(real/ai-generated):"

# --- Potentially other configurations ---
ZERO_PAD_SCORE_TABLES = True
# You can also add other global settings here if needed later.