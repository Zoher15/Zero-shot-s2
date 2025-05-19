import os
import pandas as pd
import json
import io
import re
import numpy as np
import sys
from collections import defaultdict
from pathlib import Path
import logging

# --- Project Setup ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
import config
from utils import helpers # Import your helpers module

# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Ensure logger is configured
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# --- Configuration (from original script, can be moved to config.py or a specific table_config.py if desired) ---

# Models, Methods, and Datasets
MODELS_ABBR = ['qwen2.5', 'llama3.2','CoDE'] # As per original
MODEL_NAME_MAP_FULL = {'llama3.2': 'llama3-11b', 'qwen2.5': 'qwen25-7b'}

LLM_METHODS = ['zeroshot', 'zeroshot-cot', 'zeroshot-2-artifacts']
DATASETS_TO_PROCESS = ['d3', 'df40', 'genimage'] # Renamed from DATASETS to avoid conflict

# Subset Naming (Keep as is, or move to a shared config if used by multiple table scripts)
SUBSET_NAME_MAPPING = {
    "real": "real", "image_gen0": "dfif", "image_gen1": "sd1.4", "image_gen2": "sd2.1", "image_gen3": "sdxl",
    "collabdiff": "cdif", "midjourney": "midj", "stargan": "sgan1", "starganv2": "sgan2", "styleclip": "sclip",
    "whichfaceisreal": "wfir", "adm": "adm", "biggan": "bgan", "glide_copy": "glide",
    "stable_diffusion_v_1_4": "sd1.4", "stable_diffusion_v_1_5": "sd1.5", "vqdm": "vqdm", "wukong": "wk",
    # Add any other mappings that might arise from D3 filenames if the helper's D3 loader normalizes them
    "unknown_d3_subset": "Unknown D3" # Example if load_d3_data_mapping produces this
}

# Method Name Mapping (for LLMs)
METHOD_DISPLAY_NAME_MAPPING = { # Renamed from METHOD_NAME_MAPPING
    'zeroshot': 'zero-shot',
    'zeroshot-cot': 'zero-shot-cot',
    'zeroshot-2-artifacts': r'zero-shot-s$^2$' # Display name for zeroshot-2-artifacts
}
# METHOD_TO_ADD_OURS_SUFFIX = 'zeroshot-2-artifacts' # Not directly used in this script's new flow for suffix
METHODS_TO_SHOW_DIFF = ['zeroshot-2-artifacts']
BASELINE_METHOD_FOR_DIFF = 'zeroshot-cot'

# LaTeX Formatting Options
BOLD_MAX_RECALL_PER_MODEL = True
ZERO_PAD_RECALL = True # For helpers.format_score_for_display

# Dataset-Specific Font Sizes (can be moved to a shared config or plotting config)
DEFAULT_TABLE_FONT_SIZE = r"\footnotesize"
DEFAULT_DIFF_FONT_SIZE = r"\fontsize{5}{4}\selectfont" # Raw LaTeX for font size

DATASET_TABLE_FONT_SIZES = {
    'd3':       r"\small",
    'df40':     r"\footnotesize",
    'genimage': r"\scriptsize",
}
DATASET_DIFF_FONT_SIZES = {
    'd3':       r"\tiny",
    'df40':     r"\tiny",
    'genimage': r"\fontsize{5}{5}\selectfont",
}

# CoDe specific identifiers
CODE_MODEL_ABBR = "CoDE" # To identify CoDE in MODELS_ABBR
CODE_RECALL_METHOD_KEY = "recall_data" # Internal key for CoDE's data
CODE_DISPLAY_METHOD_NAME = "trained on D3"

# --- Custom Formatting Function (kept local if specific diff formatting is needed beyond helpers.format_score_for_display) ---
def format_recall_value_with_diff(score_val, diff_val=None, diff_font_size_latex_cmd=None, zero_pad=True):
    """
    Formats recall score, optionally with a difference value.
    Uses helpers.format_score_for_display for the main score part.
    """
    # Use helper for the main score formatting
    formatted_score_str = helpers.format_score_for_display(score_val, zero_pad=zero_pad, decimal_places=1)

    if diff_val is not None and pd.notna(diff_val) and diff_font_size_latex_cmd:
        try:
            diff_float = float(diff_val)
            # Format diff with sign and 1 decimal place
            formatted_diff_str = f"{diff_float:+.1f}"
            # Append the difference string with its specific font size
            formatted_score_str += f"{{\\ {diff_font_size_latex_cmd}({formatted_diff_str})}}"
        except ValueError:
            logger.warning(f"Could not format diff value: {diff_val}")
            pass # Ignore invalid diff values, just return the formatted score
    return formatted_score_str


# --- Main Processing Logic ---
def main():
    # --- 1. Load Dataset Mappings ---
    logger.info("Loading dataset image-to-subset mappings using helpers...")
    dataset_mappings = {
        'df40': helpers.load_df40_data_mapping(config.DF40_10K_CSV_FILE),
        'd3': helpers.load_d3_data_mapping(config.D3_DIR), # Helper handles D3 structure
        'genimage': helpers.load_genimage_data_mapping(config.GENIMAGE_10K_CSV_FILE)
    }
    dataset_mappings = {k: v for k, v in dataset_mappings.items() if v and k in DATASETS_TO_PROCESS}
    if not dataset_mappings:
        logger.error("No dataset mappings loaded successfully. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded mappings for datasets: {list(dataset_mappings.keys())}")

    # --- 2. Initialize Data Structures ---
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'subsets': defaultdict(lambda: np.nan), # Stores final recall scores
        'raw_counts': defaultdict(lambda: {'TP': 0, 'FN': 0}) # Stores TP/FN from helper
    })))
    active_subsets_per_dataset = defaultdict(set) # Tracks unique subsets found

    # --- 3. Process Recall Data from Response Files using Helper ---
    logger.info("Processing .jsonl response files for TP/FN counts using helpers...")
    files_processed_count = 0
    files_missing_count = 0

    for model_abbr in MODELS_ABBR:
        for dataset_name_key in DATASETS_TO_PROCESS: # Use the keys from DATASETS_TO_PROCESS
            if dataset_name_key not in dataset_mappings:
                logger.warning(f"No mapping found for dataset '{dataset_name_key}'. Skipping recall processing.")
                continue
            current_dataset_map = dataset_mappings[dataset_name_key]

            methods_to_process_for_model = []
            filename_template_str = "" # Can be a full name or a template

            if model_abbr == CODE_MODEL_ABBR:
                methods_to_process_for_model = [CODE_RECALL_METHOD_KEY]
                # Filename for CoDE (adjust prefix if needed, e.g. AI_qwen or AI_dev)
                # Assuming eval scripts save as "AI_qwen-dataset-CoDE-rationales.jsonl"
                # Or if it's just "dataset-CoDE-rationales.jsonl"
                filename_template_str = f"AI_qwen-{dataset_name_key}-CoDE-rationales.jsonl" # Example
            else:
                model_full_name = MODEL_NAME_MAP_FULL.get(model_abbr)
                if not model_full_name:
                    logger.warning(f"No full model name for LLM '{model_abbr}'. Skipping for {dataset_name_key}.")
                    continue
                methods_to_process_for_model = LLM_METHODS
                prefix = "AI_llama" if "llama" in model_abbr.lower() else "AI_qwen"
                # Filename for LLMs (eval scripts don't add -wait0 typically)
                filename_template_str = f"{prefix}-{dataset_name_key}-{model_full_name}-{{method}}-n1-rationales.jsonl"


            for method_key_internal in methods_to_process_for_model:
                actual_file_name = ""
                if model_abbr == CODE_MODEL_ABBR:
                    actual_file_name = filename_template_str # Already full name
                else:
                    actual_file_name = filename_template_str.format(method=method_key_internal)

                rationale_file_path = config.RESPONSES_DIR / actual_file_name

                if not rationale_file_path.exists():
                    logger.info(f"Rationale file not found: {rationale_file_path}")
                    files_missing_count += 1
                    continue

                is_d3 = (dataset_name_key == 'd3')
                logger.info(f"Processing file: {rationale_file_path} (is_d3={is_d3})")

                # Use the new helper function
                subset_tp_fn_counts = helpers.get_subset_recall_counts_from_rationales(
                    rationale_file_path=rationale_file_path,
                    dataset_image_to_subset_map=current_dataset_map,
                    dataset_name_for_logging=f"{model_abbr}_{method_key_internal}_{dataset_name_key}",
                    is_d3_dataset=is_d3
                )

                if subset_tp_fn_counts:
                    # Store raw counts directly
                    all_results[dataset_name_key][model_abbr][method_key_internal]['raw_counts'] = defaultdict(
                        lambda: {'TP': 0, 'FN': 0}, subset_tp_fn_counts
                    )
                    active_subsets_per_dataset[dataset_name_key].update(subset_tp_fn_counts.keys())
                    files_processed_count += 1
                else:
                    logger.warning(f"No recall counts returned by helper for {rationale_file_path}")

    logger.info(f"\nRecall File Processing Summary:")
    logger.info(f"  Files Successfully Processed for TP/FN counts: {files_processed_count}")
    logger.info(f"  Files Missing/Not Found: {files_missing_count}")

    # --- 4. Calculate Recall Scores ---
    logger.info("\nCalculating recall scores from TP/FN counts...")
    for dataset_name_key in all_results:
        for model_abbr_key in all_results[dataset_name_key]:
            for method_key_internal in all_results[dataset_name_key][model_abbr_key]:
                raw_counts_data = all_results[dataset_name_key][model_abbr_key][method_key_internal]['raw_counts']
                if not raw_counts_data:
                    logger.warning(f"No raw_counts found for {dataset_name_key}/{model_abbr_key}/{method_key_internal} to calculate recall.")
                    continue
                for subset_name_key, counts_dict in raw_counts_data.items():
                    tp = counts_dict.get('TP', 0)
                    fn = counts_dict.get('FN', 0)
                    recall_score = helpers.get_recall(tp, fn) # Use helper
                    # Store recall score, rounded to 1 decimal place as per original formatting
                    all_results[dataset_name_key][model_abbr_key][method_key_internal]['subsets'][subset_name_key] = round(recall_score * 100, 1)

    # --- 5. Generate LaTeX Tables ---
    logger.info("\nGenerating LaTeX tables...")
    final_latex_string_io = io.StringIO()

    # Add suggested LaTeX preamble comments
    final_latex_string_io.write("% --- Suggested LaTeX Preamble ---\n")
    final_latex_string_io.write("% \\usepackage{booktabs, multirow, graphicx, caption, amsmath, siunitx, amsfonts}\n")
    final_latex_string_io.write("% \\begin{document}\n\n")

    sorted_dataset_keys = sorted(list(dataset_mappings.keys()))

    for dataset_name_key in sorted_dataset_keys:
        if dataset_name_key not in all_results or not all_results[dataset_name_key]:
            logger.info(f"Skipping table for dataset '{dataset_name_key}' as no results were found.")
            continue

        logger.info(f"Generating table for dataset: {dataset_name_key}")
        dataset_level_results_dict = all_results[dataset_name_key]
        current_active_subsets_raw_keys = sorted(list(active_subsets_per_dataset.get(dataset_name_key, set())))

        # --- Subset Name Mapping and Sorting for Table Columns ---
        sorted_subsets_for_display_order_raw = []
        if not current_active_subsets_raw_keys:
            logger.warning(f"No active subsets found for '{dataset_name_key}'. Table columns might be empty.")
        else:
            # Ensure all active raw subsets have a display mapping
            for s_raw in current_active_subsets_raw_keys:
                if s_raw not in SUBSET_NAME_MAPPING:
                    s_lower = s_raw.lower()
                    if s_lower not in SUBSET_NAME_MAPPING.values():
                         SUBSET_NAME_MAPPING[s_raw] = s_lower # Default to lowercase if no specific mapping
                         logger.info(f"Added default display mapping for subset '{s_raw}' -> '{s_lower}'")
                    elif SUBSET_NAME_MAPPING.get(s_raw) is None:
                         SUBSET_NAME_MAPPING[s_raw] = s_lower

            # Sort raw keys based on their mapped display names for column order
            mapped_display_names_for_sort = {s_raw: SUBSET_NAME_MAPPING.get(s_raw, s_raw.lower())
                                             for s_raw in current_active_subsets_raw_keys}
            temp_sorted_raw_keys = sorted(current_active_subsets_raw_keys,
                                          key=lambda s_raw: mapped_display_names_for_sort[s_raw])

            if 'real' in temp_sorted_raw_keys: # Prioritize 'real' column
                temp_sorted_raw_keys.remove('real')
                sorted_subsets_for_display_order_raw = ['real'] + temp_sorted_raw_keys
            else:
                sorted_subsets_for_display_order_raw = temp_sorted_raw_keys

        # --- Start LaTeX Table Environment ---
        final_latex_string_io.write(f"% --- Table for Dataset: {dataset_name_key} ---\n")
        final_latex_string_io.write(r"\begin{table*}[htbp]" + "\n")
        final_latex_string_io.write(r"\centering" + "\n")

        dataset_display_name_escaped = helpers.escape_latex(dataset_name_key.upper())
        baseline_method_display_escaped = helpers.escape_latex(METHOD_DISPLAY_NAME_MAPPING.get(BASELINE_METHOD_FOR_DIFF, BASELINE_METHOD_FOR_DIFF))
        s2_method_display_escaped = helpers.escape_latex(METHOD_DISPLAY_NAME_MAPPING.get('zeroshot-2-artifacts', 'zeroshot-s$^2$'))
        caption_str = f"Recall Scores for {dataset_display_name_escaped}. Scores to 1 d.p. Differences for {s2_method_display_escaped} vs {baseline_method_display_escaped} shown in parentheses."
        final_latex_string_io.write(f"\\caption{{{caption_str}}}\n")
        final_latex_string_io.write(f"\\label{{tab:{dataset_name_key}_recall_scores}}\n")

        current_table_font = DATASET_TABLE_FONT_SIZES.get(dataset_name_key, DEFAULT_TABLE_FONT_SIZE)
        current_diff_font = DATASET_DIFF_FONT_SIZES.get(dataset_name_key, DEFAULT_DIFF_FONT_SIZE)
        final_latex_string_io.write(r"\setlength{\tabcolsep}{3pt}" + "\n")
        final_latex_string_io.write(current_table_font + "\n")

        num_subset_cols = len(sorted_subsets_for_display_order_raw)
        col_spec = 'll' + ('l' * num_subset_cols if num_subset_cols > 0 else '')
        final_latex_string_io.write(r"\begin{tabular}{" + col_spec + "}\n")
        final_latex_string_io.write(r"\toprule" + "\n")

        header_cells_list = [r"\textbf{Model}", r"\textbf{Method}"]
        for subset_raw_key_header in sorted_subsets_for_display_order_raw:
            display_col_name = SUBSET_NAME_MAPPING.get(subset_raw_key_header, subset_raw_key_header.lower())
            header_cells_list.append(rf"\textbf{{{helpers.escape_latex(display_col_name)}}}")
        final_latex_string_io.write(" & ".join(header_cells_list) + r" \\" + "\n")
        final_latex_string_io.write(r"\midrule" + "\n")

        # --- Table Body ---
        models_in_table = [m for m in MODELS_ABBR if m in dataset_level_results_dict and dataset_level_results_dict[m]]

        for model_idx, model_abbr_render in enumerate(models_in_table):
            model_data_for_table = dataset_level_results_dict[model_abbr_render]
            
            methods_for_current_model = []
            if model_abbr_render == CODE_MODEL_ABBR:
                if CODE_RECALL_METHOD_KEY in model_data_for_table:
                    methods_for_current_model = [CODE_RECALL_METHOD_KEY]
            else:
                methods_for_current_model = [m for m in LLM_METHODS if m in model_data_for_table]

            if not methods_for_current_model:
                logger.info(f"Skipping {model_abbr_render} in {dataset_name_key} table: no method data found.")
                continue

            max_recall_per_subset_for_model = defaultdict(lambda: -1.0)
            baseline_scores_for_diff = {}

            if BOLD_MAX_RECALL_PER_MODEL and num_subset_cols > 0 and model_abbr_render != CODE_MODEL_ABBR:
                for method_k_bold in methods_for_current_model:
                    subset_scores = model_data_for_table[method_k_bold].get('subsets', {})
                    for subset_raw_k_bold in sorted_subsets_for_display_order_raw:
                        recall_val = subset_scores.get(subset_raw_k_bold, np.nan)
                        if pd.notna(recall_val) and recall_val > max_recall_per_subset_for_model[subset_raw_k_bold]:
                            max_recall_per_subset_for_model[subset_raw_k_bold] = recall_val
            
            if model_abbr_render != CODE_MODEL_ABBR and BASELINE_METHOD_FOR_DIFF in methods_for_current_model:
                baseline_scores_for_diff = model_data_for_table[BASELINE_METHOD_FOR_DIFF].get('subsets', {})

            num_methods_for_rowspan = len(methods_for_current_model)
            for method_row_idx, method_key_render in enumerate(methods_for_current_model):
                row_cells_list = []
                model_display_escaped = helpers.escape_latex(model_abbr_render)
                if num_methods_for_rowspan == 1:
                    row_cells_list.append(model_display_escaped)
                elif method_row_idx == 0:
                    row_cells_list.append(f"\\multirow{{{num_methods_for_rowspan}}}{{*}}{{{model_display_escaped}}}")
                else:
                    row_cells_list.append("")

                method_display_name = ""
                if model_abbr_render == CODE_MODEL_ABBR:
                    method_display_name = helpers.escape_latex(CODE_DISPLAY_METHOD_NAME)
                else:
                    method_display_name = helpers.escape_latex(METHOD_DISPLAY_NAME_MAPPING.get(method_key_render, method_key_render))
                row_cells_list.append(method_display_name)

                current_method_subset_scores = model_data_for_table[method_key_render].get('subsets', {})
                for subset_raw_key_cell in sorted_subsets_for_display_order_raw:
                    recall_value = current_method_subset_scores.get(subset_raw_key_cell, np.nan)
                    diff_value_for_cell = np.nan
                    if model_abbr_render != CODE_MODEL_ABBR and method_key_render in METHODS_TO_SHOW_DIFF:
                        baseline_recall = baseline_scores_for_diff.get(subset_raw_key_cell, np.nan)
                        if pd.notna(recall_value) and pd.notna(baseline_recall):
                            diff_value_for_cell = recall_value - baseline_recall
                    
                    formatted_cell_value = format_recall_value_with_diff(
                        recall_value,
                        diff_val=diff_value_for_cell if pd.notna(diff_value_for_cell) else None,
                        diff_font_size_latex_cmd=current_diff_font,
                        zero_pad=ZERO_PAD_RECALL
                    )

                    current_max_recall = max_recall_per_subset_for_model.get(subset_raw_key_cell, -1.0)
                    if BOLD_MAX_RECALL_PER_MODEL and model_abbr_render != CODE_MODEL_ABBR and \
                       pd.notna(recall_value) and np.isclose(recall_value, current_max_recall):
                        
                        diff_marker = f"{{\\ {current_diff_font}(" # Start of diff part
                        if diff_marker in formatted_cell_value:
                            parts = formatted_cell_value.split(diff_marker, 1)
                            formatted_cell_value = f"\\textbf{{{parts[0]}}}{diff_marker}{parts[1]}"
                        else:
                            formatted_cell_value = f"\\textbf{{{formatted_cell_value}}}"
                    row_cells_list.append(formatted_cell_value)
                
                final_latex_string_io.write(" & ".join(row_cells_list) + r" \\" + "\n")

            if model_idx < len(models_in_table) - 1:
                final_latex_string_io.write(r"\midrule" + "\n")

        final_latex_string_io.write(r"\bottomrule" + "\n")
        final_latex_string_io.write(r"\end{tabular}" + "\n")
        final_latex_string_io.write(r"\end{table*}" + "\n\n")

    final_latex_string_io.write("% --- End of Generated Tables ---\n")
    final_latex_string_io.write("% \\end{document}\n")

    # --- 7. Output ---
    final_latex_output = final_latex_string_io.getvalue()
    final_latex_string_io.close()

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_tex_filename = "recall_subsets_tables_refactored.tex" # New name to avoid overwrite
    output_tex_filepath = config.TABLES_DIR / output_tex_filename

    try:
        with open(output_tex_filepath, 'w', encoding='utf-8') as f:
            f.write(final_latex_output)
        logger.info(f"LaTeX table code saved to: {output_tex_filepath}")
    except IOError as e:
        logger.error(f"Error saving LaTeX to file {output_tex_filepath}: {e}", exc_info=True)

    logger.info("\nNotes for Refactored Script:")
    logger.info("- Uses helpers for data loading, subset recall TP/FN counting, and LaTeX formatting.")
    logger.info(f"- Bolding ({BOLD_MAX_RECALL_PER_MODEL=}) applies only to max LLM scores per subset.")
    logger.info(f"- Differences shown for '{s2_method_display_escaped}' vs '{baseline_method_display_escaped}'.")
    logger.info(f"- Recall scores/differences formatted to 1 decimal place ({ZERO_PAD_RECALL=}).")

if __name__ == "__main__":
    main()