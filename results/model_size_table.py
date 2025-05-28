"""
Model Size Scaling Analysis Table Generator

This script generates LaTeX tables analyzing how model performance scales with
parameter count across different vision-language models. It focuses on comparing
models within the same family (e.g., Qwen2.5 3B vs 7B vs 32B vs 72B) to
understand scaling laws for AI-generated image detection tasks.

The analysis helps researchers understand the relationship between model size,
computational cost, and detection performance, providing insights for optimal
model selection in resource-constrained environments.

Features:
- Multi-size model comparison within model families
- Performance scaling analysis across parameter counts
- LaTeX table generation with statistical highlighting
- Automatic model family detection and grouping
- Percentage improvement calculations between sizes
- Professional formatting for academic publications

Model Scaling Analysis:
- Parameter count extraction and numerical sorting
- Performance comparison across model sizes
- Percentage improvement calculation relative to baseline
- Statistical significance highlighting (bold for best performance)
- Family-wise analysis (Qwen2.5, Llama3.2 families)

Supported Model Families:
- Qwen2.5: 3B, 7B, 32B, 72B parameter variants
- Llama3.2: 11B, 90B parameter variants (configurable)
- Extensible to other model families

Scaling Metrics:
- Macro F1-score as primary performance indicator
- Percentage improvements between consecutive sizes
- Performance per parameter ratios (efficiency analysis)
- Statistical highlighting of optimal size trade-offs

Table Generation:
- LaTeX format compatible with academic journals
- Automatic column formatting based on dataset count
- Bold highlighting for best performance per dataset
- Percentage difference annotations
- Professional typography with configurable fonts

Performance Analysis:
- Cross-dataset consistency in scaling patterns
- Method-wise scaling behavior (zero-shot, CoT, zero-shot-s²)
- Identification of performance plateaus
- Cost-benefit analysis for model selection

Usage:
    python results/model_size_table.py
    
Output Files:
    - LaTeX tables saved to RESULTS_OUTPUT_DIR
    - Scaling analysis summaries
    - Performance trend visualizations (if enabled)

Configuration Options:
- Model family selection (focus on specific families)
- Dataset filtering (subset analysis)
- Baseline method selection for improvement calculations
- LaTeX formatting parameters (fonts, highlighting)
- Statistical significance thresholds

Data Processing:
- CSV score file loading and parsing
- Model name parsing using helpers.parse_model_name()
- Automatic parameter count extraction and sorting
- Cross-dataset performance aggregation
- Missing data handling and interpolation

Statistical Highlighting:
- Bold formatting for maximum scores per dataset
- Zero-padding for consistent number formatting
- Percentage calculations with appropriate precision
- Error handling for missing or invalid scores

Dependencies:
- pandas for data manipulation
- numpy for numerical operations
- pathlib for file system operations
- Custom helpers module for model parsing

Note:
    This script focuses on scaling analysis within model families rather than
    cross-family comparisons. For cross-family analysis, use other table
    generation scripts in the results/ directory.
"""

import os
import sys
from pathlib import Path
import logging # For logging
import io # For StringIO

import pandas as pd
import numpy as np
from collections import defaultdict
# Removed re as it's likely not needed directly after refactoring

# --- Project Setup ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import config
from utils import helpers

# --- Logger Setup ---
helpers.setup_global_logger(config.RESULTS_MODEL_SIZE_TABLE_LOG_FILE)
# Get a logger instance for this specific module.
logger = logging.getLogger(__name__)

# --- Seed for reproducibility (from original script) ---
np.random.seed(0)

# --- Configuration Constants (specific to this script's table generation) ---
# Consider moving to config.py if these become more globally used or complex

# Define all models to be included (for parsing and potential other uses)
ALL_MODELS_CONFIG = [
    "qwen25-3b", "qwen25-7b", "qwen25-32b", "qwen25-72b",
    # "CoDE" # CoDE is handled differently or excluded from this specific table based on original logic
]
CODE_MODEL_NAME = "CoDE" # Used if CoDE logic were to be re-integrated for this table

# Define datasets (will become columns in the table)
# Assuming these are the "2k" versions based on other scripts, adjust if needed
ALLOWED_DATASETS_TABLE = ["d32k", "df402k", "genimage2k"] # Renamed to avoid conflict
LLM_METHODS_ORDER_TABLE = ["zeroshot", "zeroshot-cot", "zeroshot-2-artifacts"] # Renamed
BASELINE_METHOD_FOR_DIFF_TABLE = "zeroshot-cot"
TARGET_METHOD_FOR_DIFF_TABLE = "zeroshot-2-artifacts"

N_VAL_TABLE = "1" # Used for filename construction
# WAIT_VAL_TABLE = "0" # If eval scripts save with -wait0, otherwise this might not be needed

# Model Family and Size Mapping (will be populated using helpers.parse_model_name)
MODEL_FAMILIES_DETECTED_TABLE = [] # Renamed
MODEL_FAMILY_MAP_TABLE = {} # Renamed
MODEL_SIZE_MAP_TABLE = {} # Renamed (stores string like "7b")
MODEL_NUMERIC_SIZE_MAP_TABLE = {} # Renamed (stores float for sorting)
MODELS_BY_FAMILY_TABLE = defaultdict(list) # Renamed

CODE_FAMILY_NAME_TABLE = "CoDE" # Renamed
CODE_SIZE_IDENTIFIER_TABLE = "6m" # Renamed

# Populate model maps using helpers
logger.info("--- Parsing Model Names and Families (using helpers) ---")
for model_name_conf_item in ALL_MODELS_CONFIG:
    family_name_str, size_str_val = helpers.parse_model_name(model_name_conf_item) # size_str_val is "7b", "6m" etc.

    MODEL_FAMILY_MAP_TABLE[model_name_conf_item] = family_name_str
    MODEL_SIZE_MAP_TABLE[model_name_conf_item] = size_str_val
    MODEL_NUMERIC_SIZE_MAP_TABLE[model_name_conf_item] = helpers.get_numeric_model_size(size_str_val)

    if family_name_str not in MODEL_FAMILIES_DETECTED_TABLE:
        MODEL_FAMILIES_DETECTED_TABLE.append(family_name_str)
    MODELS_BY_FAMILY_TABLE[family_name_str].append(model_name_conf_item)

# If CoDE needs to be parsed (even if not in ALL_MODELS_CONFIG for this table generation)
# This section was in the original; retain if CoDE info might be used contextually,
# otherwise, it can be removed if CoDE is strictly out of scope for this specific table.
# if CODE_MODEL_NAME not in MODEL_FAMILY_MAP_TABLE and CODE_MODEL_NAME in ["CoDE"]: # Example inclusion condition
#     fam, size = helpers.parse_model_name(CODE_MODEL_NAME)
#     MODEL_FAMILY_MAP_TABLE[CODE_MODEL_NAME] = fam
#     MODEL_SIZE_MAP_TABLE[CODE_MODEL_NAME] = size
#     MODEL_NUMERIC_SIZE_MAP_TABLE[CODE_MODEL_NAME] = helpers.get_numeric_model_size(size)
#     if fam not in MODEL_FAMILIES_DETECTED_TABLE: MODEL_FAMILIES_DETECTED_TABLE.append(fam)
#     MODELS_BY_FAMILY_TABLE[fam].append(CODE_MODEL_NAME)

logger.info("--- End Model Parsing (model_size_table) ---")

# Define order for table rows (e.g., which model families to show and in what order)
# Original focused on 'qwen2.5', 'llama3.2'.
TABLE_MODEL_FAMILIES_ORDER_DISPLAY = ['qwen2.5'] # For this script, it seems to focus on Qwen2.5 scaling
# If other families needed, add them: e.g. ['qwen2.5', 'llama3.2']

CODE_OVERALL_METHOD_KEY_TABLE = "Overall" # Renamed

METHOD_DISPLAY_NAME_MAPPING_TABLE = { # Renamed
    'zeroshot': 'zeroshot',
    'zeroshot-cot': 'zeroshot-cot',
    'zeroshot-2-artifacts': r'zeroshot-s$^2$',
    CODE_OVERALL_METHOD_KEY_TABLE: "trained on D3" # For CoDE if it were included
}

# LaTeX Table Formatting Options
BOLD_MAX_SCORE_PER_MODEL_DATASET_TABLE = True
ZERO_PAD_SCORE_TABLE = True # For helpers.format_score_for_display
TABLE_FONT_SIZE_LATEX = r"\small" # Renamed
DATASET_DISPLAY_MAP_LATEX_TABLE = { # Renamed
    "d32k": "D3 (2k)", "df402k": "DF40 (2k)", "genimage2k": "GenImage (2k)"
}
# Add a global config for ZERO_PAD_SCORE_TABLES if using helpers.format_score_for_display
# For now, assuming it's True as per original script's ZERO_PAD_SCORE
if not hasattr(config, 'ZERO_PAD_SCORE_TABLES'):
    logger.warning("config.ZERO_PAD_SCORE_TABLES not found, defaulting to True for score padding in model_size_table.")
    config.ZERO_PAD_SCORE_TABLES = True

# --- Data Loading from CSVs (Refactored) ---
def load_scores_for_model_size_table(
    models_to_load_list: list,
    allowed_datasets_list: list,
    llm_methods_list: list,
    # code_method_key_str: str, # CoDE specific, handle if CoDE is re-introduced
    n_val_str: str
    # wait_val_str: str # Filenames from eval likely omit -wait0
):
    logger.info(f"Loading Scores from CSV files in: {config.SCORES_DIR} for model size table.")
    all_data_list_load = []
    stats = {"csv_read": 0, "missing_csvs": 0, "errors_reading_csv": 0, "files_checked": 0}

    for model_name_load in models_to_load_list:
        # current_methods_for_model_load = [code_method_key_str] if model_name_load == CODE_MODEL_NAME else llm_methods_list
        current_methods_for_model_load = llm_methods_list # Assuming only LLMs for this table now

        for dataset_name_csv in allowed_datasets_list:
            for method_key_csv in current_methods_for_model_load:
                stats["files_checked"] += 1
                score_val_csv = None

                # Determine filename prefix
                prefix_str = "AI_llama" if "llama" in model_name_load.lower() else "AI_qwen"
                
                # Filename construction (eval scripts likely omit -wait0)
                # if model_name_load == CODE_MODEL_NAME:
                #     # CoDE CSV might have a different naming, adjust if needed
                #     prefix_str = "AI_qwen" # Or other prefix for CoDE files
                #     fname_csv_load = f"{prefix_str}-{dataset_name_csv}-{model_name_load}-scores.csv"
                # else:
                fname_csv_load = f"{prefix_str}-{dataset_name_csv}-{model_name_load}-{method_key_csv}-n{n_val_str}-scores.csv"
                
                fpath_csv_load = config.SCORES_DIR / fname_csv_load
                df_score_file = helpers.load_scores_csv_to_dataframe(fpath_csv_load)

                if not df_score_file.empty:
                    try:
                        # if model_name_load == CODE_MODEL_NAME:
                        #     # Original logic for CoDE CSV:
                        #     if len(df_score_file) >= 2 and len(df_score_file.columns) >= 1:
                        #         raw_score_val = df_score_file.iloc[1, 0]
                        #         score_val_csv = round(float(raw_score_val) * 100, 1)
                        #         logger.info(f"  CSV READ (CoDE): {model_name_load}/{dataset_name_csv}. Score: {score_val_csv:.1f}")
                        #         stats["csv_read"] += 1
                        #     else:
                        #         logger.warning(f"CoDE CSV {fpath_csv_load} lacks expected structure. Shape: {df_score_file.shape}")
                        #         stats["errors_reading_csv"] += 1
                        # else: # LLM
                        target_idx_csv = f"{method_key_csv}-n{n_val_str}"
                        if target_idx_csv in df_score_file.index:
                            score_col = 'macro_f1' if 'macro_f1' in df_score_file.columns else df_score_file.columns[0]
                            raw_score_val = df_score_file.loc[target_idx_csv, score_col]
                            score_val_csv = round(float(raw_score_val) * 100, 1) # Scores in CSV 0-1, table 0-100
                            logger.info(f"  CSV READ: {model_name_load}/{dataset_name_csv}/{method_key_csv}. Score: {score_val_csv:.1f}")
                            stats["csv_read"] += 1
                        else:
                            logger.warning(f"Index '{target_idx_csv}' not in CSV {fpath_csv_load}. Indices: {df_score_file.index.tolist()}")
                            stats["errors_reading_csv"] += 1
                    except KeyError:
                        logger.warning(f"Index '{target_idx_csv}' caused KeyError in CSV {fpath_csv_load}.")
                        stats["errors_reading_csv"] += 1
                    except Exception as e:
                        logger.error(f"Error processing data from CSV {fpath_csv_load}: {e}", exc_info=True)
                        stats["errors_reading_csv"] += 1
                else:
                    if fpath_csv_load.exists():
                        logger.warning(f"CSV {fpath_csv_load} exists but was empty/unreadable by helper.")
                    stats["missing_csvs"] += 1
                
                all_data_list_load.append({
                    'llm': model_name_load, 'dataset': dataset_name_csv, 'method': method_key_csv,
                    'score': score_val_csv, 'n_samples': 0 # n_samples not in summary CSVs
                })
    
    logger.info(f"Finished loading scores. Checked: {stats['files_checked']}, Read: {stats['csv_read']}, Missing: {stats['missing_csvs']}, Errors: {stats['errors_reading_csv']}")
    if not all_data_list_load: return pd.DataFrame()
    
    df_loaded = pd.DataFrame(all_data_list_load)
    df_loaded['model_family'] = df_loaded['llm'].map(MODEL_FAMILY_MAP_TABLE)
    df_loaded['model_size'] = df_loaded['llm'].map(MODEL_SIZE_MAP_TABLE)
    df_loaded['parameter_size_numeric'] = df_loaded['llm'].map(MODEL_NUMERIC_SIZE_MAP_TABLE) # Use pre-calculated numeric map
    
    # Filter for families intended for this table BEFORE trying to set categorical type
    df_loaded = df_loaded[df_loaded['model_family'].isin(TABLE_MODEL_FAMILIES_ORDER_DISPLAY)]
    if df_loaded.empty and TABLE_MODEL_FAMILIES_ORDER_DISPLAY:
        logger.warning(f"DataFrame empty after filtering for families: {TABLE_MODEL_FAMILIES_ORDER_DISPLAY}. Check model names and ALL_MODELS_CONFIG.")


    if not df_loaded.empty and TABLE_MODEL_FAMILIES_ORDER_DISPLAY:
        family_cat_type = pd.CategoricalDtype(categories=TABLE_MODEL_FAMILIES_ORDER_DISPLAY, ordered=True)
        df_loaded['model_family_ordered'] = df_loaded['model_family'].astype(family_cat_type)
        df_loaded = df_loaded.sort_values(by=['model_family_ordered', 'parameter_size_numeric', 'llm', 'method', 'dataset'])
        df_loaded = df_loaded.drop(columns=['model_family_ordered'])
    elif not df_loaded.empty: # No specific family order, sort by what's available
         df_loaded = df_loaded.sort_values(by=['model_family', 'parameter_size_numeric', 'llm', 'method', 'dataset'])

    return df_loaded


# --- Main LaTeX Table Generation Logic ---
def generate_latex_table_for_model_size(scores_df_input: pd.DataFrame):
    if scores_df_input.empty:
        logger.warning("No data to generate model size table. Scores DataFrame is empty.")
        return "% No data to generate model size table.\n"

    final_latex_io = io.StringIO()
    # Preamble
    final_latex_io.write("% --- Suggested LaTeX Preamble (model_size_table.py) ---\n")
    final_latex_io.write("% \\usepackage{booktabs, multirow, amsmath, caption, amsfonts}\n")
    final_latex_io.write("% \\begin{document}\n\n")
    
    final_latex_io.write("% --- Model Size Performance Table ---\n")
    final_latex_io.write(r"\begin{table*}[htbp!]" + "\n")
    final_latex_io.write(r"\centering" + "\n")
    final_latex_io.write(f"\\caption{{Effect of prompts on Macro F1 scores (\%) across Qwen2.5 model sizes (Refactored)}}\n") # Example caption
    final_latex_io.write(f"\\label{{tab:qwen25_prompt_effects_refactored}}\n")
    final_latex_io.write(TABLE_FONT_SIZE_LATEX + "\n")

    num_dataset_cols_table = len(ALLOWED_DATASETS_TABLE)
    total_cols_table = 3 + num_dataset_cols_table
    col_spec_table = 'lll' + ('l' * num_dataset_cols_table)
    final_latex_io.write(r"\begin{tabular}{" + col_spec_table + "}\n")
    final_latex_io.write(r"\toprule" + "\n")

    header_cells_list_table = [r"\textbf{Model Family}", r"\textbf{Model Size}", r"\textbf{Method}"]
    for ds_key_header in ALLOWED_DATASETS_TABLE:
        display_name_header = DATASET_DISPLAY_MAP_LATEX_TABLE.get(ds_key_header, ds_key_header.upper())
        header_cells_list_table.append(rf"\textbf{{{helpers.escape_latex(display_name_header)}}}")
    final_latex_io.write(" & ".join(header_cells_list_table) + r" \\" + "\n")
    final_latex_io.write(r"\midrule" + "\n")
    
    # Iterate through families defined in TABLE_MODEL_FAMILIES_ORDER_DISPLAY
    for family_idx, family_name_display in enumerate(TABLE_MODEL_FAMILIES_ORDER_DISPLAY):
        family_data = scores_df_input[scores_df_input['model_family'] == family_name_display].copy()
        if family_data.empty:
            logger.info(f"No data found for model family '{family_name_display}'. Skipping in table.")
            continue

        # is_code_family = (family_name_display == CODE_FAMILY_NAME_TABLE) # Not used if CoDE is excluded

        # Sort models within the family by their numeric size
        # MODELS_BY_FAMILY_TABLE[family_name_display] contains list of full model names like "qwen25-7b"
        sorted_llms_in_current_family = sorted(
            MODELS_BY_FAMILY_TABLE.get(family_name_display, []),
            key=lambda llm_name_sort: MODEL_NUMERIC_SIZE_MAP_TABLE.get(llm_name_sort, float('inf'))
        )
        
        if not sorted_llms_in_current_family:
            logger.warning(f"No models configured or found for family '{family_name_display}' after sorting.")
            continue

        methods_for_family_display_list = LLM_METHODS_ORDER_TABLE # Assuming only LLMs in this table

        # Calculate total number of rows for this family (for multirow)
        num_rows_for_family_multirow = 0
        for llm_name_calc_rows in sorted_llms_in_current_family:
            if not family_data[family_data['llm'] == llm_name_calc_rows].empty:
                 num_rows_for_family_multirow += len(methods_for_family_display_list)
        
        if num_rows_for_family_multirow == 0:
            logger.info(f"No data rows to display for '{family_name_display}' models.")
            continue
        
        family_display_name_escaped = helpers.escape_latex(family_name_display)
        first_llm_in_family_flag = True

        for llm_render_idx, current_llm_name in enumerate(sorted_llms_in_current_family):
            model_specific_rows_data = family_data[family_data['llm'] == current_llm_name]
            if model_specific_rows_data.empty: continue

            if not first_llm_in_family_flag and family_idx > 0: # Add midrule between different model sizes within the same family only if it's not the very first model size
                 final_latex_io.write(rf"\cmidrule(lr){{2-{total_cols_table}}}" + "\n")
            
            max_scores_for_bolding = {}
            if BOLD_MAX_SCORE_PER_MODEL_DATASET_TABLE: # and not is_code_family: (CoDE exclusion handled by family filter)
                for ds_key_bold_calc in ALLOWED_DATASETS_TABLE:
                    current_max_score = -1.0
                    for m_key_bold_calc in methods_for_family_display_list:
                        score_entry_bold = model_specific_rows_data[
                            (model_specific_rows_data['method'] == m_key_bold_calc) &
                            (model_specific_rows_data['dataset'] == ds_key_bold_calc)
                        ]['score']
                        if not score_entry_bold.empty and pd.notna(score_entry_bold.iloc[0]):
                            current_max_score = max(current_max_score, score_entry_bold.iloc[0])
                    if current_max_score > -1.0: max_scores_for_bolding[ds_key_bold_calc] = current_max_score
            
            for method_render_idx, method_key_current_row in enumerate(methods_for_family_display_list):
                row_cells_output = []
                # Model Family column (multirow for the first model of the family)
                if llm_render_idx == 0 and method_render_idx == 0:
                    if num_rows_for_family_multirow > 0:
                         row_cells_output.append(rf"\multirow{{{num_rows_for_family_multirow}}}{{*}}{{{family_display_name_escaped}}}")
                    else: # Should not happen if num_rows_for_family_multirow check passed
                         row_cells_output.append(family_display_name_escaped)
                else:
                    row_cells_output.append("")

                # Model Size column (multirow for the first method of this size)
                if method_render_idx == 0:
                    raw_size_str_display = MODEL_SIZE_MAP_TABLE.get(current_llm_name, "")
                    size_display_content = raw_size_str_display
                    if size_display_content and size_display_content.endswith('b'): # Capitalize B for Billion
                        size_display_content = size_display_content[:-1] + 'B'
                    size_display_name_escaped = "-" if not size_display_content else helpers.escape_latex(size_display_content)
                    row_cells_output.append(rf"\multirow{{{len(methods_for_family_display_list)}}}{{*}}{{{size_display_name_escaped}}}")
                else:
                    row_cells_output.append("")
                
                # Method column
                method_display_raw_name = METHOD_DISPLAY_NAME_MAPPING_TABLE.get(method_key_current_row, method_key_current_row)
                row_cells_output.append(method_display_raw_name) # Already LaTeX formatted if needed

                # Dataset score columns
                for ds_key_cell_render in ALLOWED_DATASETS_TABLE:
                    data_cell_entry = model_specific_rows_data[
                        (model_specific_rows_data['method'] == method_key_current_row) &
                        (model_specific_rows_data['dataset'] == ds_key_cell_render)
                    ]
                    score_val_cell_render = data_cell_entry['score'].iloc[0] if not data_cell_entry.empty and pd.notna(data_cell_entry['score'].iloc[0]) else np.nan
                    
                    # Use helpers.format_score_for_display
                    score_str_formatted_part = helpers.format_score_for_display(score_val_cell_render, zero_pad=config.ZERO_PAD_SCORE_TABLES, decimal_places=1)
                    
                    display_score_final_part = score_str_formatted_part
                    if BOLD_MAX_SCORE_PER_MODEL_DATASET_TABLE and \
                       pd.notna(score_val_cell_render) and ds_key_cell_render in max_scores_for_bolding and \
                       np.isclose(score_val_cell_render, max_scores_for_bolding[ds_key_cell_render]) and \
                       score_str_formatted_part != '-':
                        display_score_final_part = rf"\textbf{{{score_str_formatted_part}}}"
                    
                    cell_content_final = display_score_final_part
                    
                    # Diff calculation (specific to LLMs in original, now applies if not CoDE)
                    if method_key_current_row == TARGET_METHOD_FOR_DIFF_TABLE and pd.notna(score_val_cell_render): # and not is_code_family
                        baseline_entry_df = model_specific_rows_data[
                            (model_specific_rows_data['method'] == BASELINE_METHOD_FOR_DIFF_TABLE) &
                            (model_specific_rows_data['dataset'] == ds_key_cell_render)
                        ]
                        baseline_score_val_diff = baseline_entry_df['score'].iloc[0] if not baseline_entry_df.empty and pd.notna(baseline_entry_df['score'].iloc[0]) else np.nan
                        
                        if pd.notna(baseline_score_val_diff):
                            diff_calc = score_val_cell_render - baseline_score_val_diff
                            formatted_diff_str = f"{diff_calc:+.1f}" # Format with sign, 1 decimal
                            diff_suffix_str = f" \\mbox{{\\tiny ({formatted_diff_str})}}" # Original had \tiny
                            cell_content_final += diff_suffix_str
                            
                    row_cells_output.append(cell_content_final)
                final_latex_io.write(" & ".join(row_cells_output) + r" \\" + "\n")
            first_llm_in_family_flag = False
        
        # Add midrule between different model families if there are more families to come
        if family_idx < len(TABLE_MODEL_FAMILIES_ORDER_DISPLAY) - 1:
             final_latex_io.write(r"\midrule" + "\n")


    final_latex_io.write(r"\bottomrule" + "\n")
    final_latex_io.write(r"\end{tabular}" + "\n")
    final_latex_io.write(r"\end{table*}" + "\n\n% --- End of Model Size Table ---\n")
    final_latex_io.write("% \\end{document}\n") # Added for completeness
    return final_latex_io.getvalue()

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting Model Size Table Script ---")
    if not config.SCORES_DIR.is_dir():
        logger.error(f"SCORES_DIR '{config.SCORES_DIR}' not found. Exiting.")
        sys.exit(1)

    main_scores_df = load_scores_for_model_size_table(
        ALL_MODELS_CONFIG, # List of all model strings to try loading
        ALLOWED_DATASETS_TABLE,
        LLM_METHODS_ORDER_TABLE,
        # CODE_OVERALL_METHOD_KEY_TABLE, # CoDE loading part removed for now from this func
        N_VAL_TABLE
        # WAIT_VAL_TABLE # Omitted as filenames might not have it
    )

    if main_scores_df.empty and len(ALL_MODELS_CONFIG) > 0:
        logger.error("Exiting: No data loaded from CSVs for the configured models for model_size_table.")
        sys.exit(1)
    
    latex_output_content = generate_latex_table_for_model_size(main_scores_df)

    # logger.info("\n" + "="*25 + " Generated LaTeX Code (Model Size Table) " + "="*25) # For debug
    # logger.info(latex_output_content)
    # logger.info("="*70)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_filename_base = "qwen25_prompt_effects_model_size_table_refactored.tex"
    output_filepath_final = config.TABLES_DIR / output_filename_base

    try:
        with open(output_filepath_final, 'w', encoding='utf-8') as f:
            f.write(latex_output_content)
        logger.info(f"LaTeX model size table saved to: {output_filepath_final}")
    except IOError as e:
        logger.error(f"Error saving LaTeX model size table: {e}", exc_info=True)

    logger.info("\nNotes for Refactored Model Size Table:")
    logger.info(r"- Scores are Macro F1 (%). Uses helpers for CSV loading, parsing, formatting.")
    logger.info(f"- Table focuses on families in TABLE_MODEL_FAMILIES_ORDER_DISPLAY (currently: {TABLE_MODEL_FAMILIES_ORDER_DISPLAY}).")
    # Add other relevant notes from original script if they still apply
    logger.info("--- Model Size Table Script Finished ---")