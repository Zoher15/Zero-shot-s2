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
helpers.setup_global_logger(config.RESULTS_COMBINE_TABLES_LOG_FILE)
# Get a logger instance for this specific module.
logger = logging.getLogger(__name__)

# --- Configuration (Adapted from original combine_tables.py) ---

# Models, Methods, and Datasets
# Note: MODELS_ABBR in combine_tables.py was ['qwen2.5', 'llama3', 'CoDE']
# while in recall_subsets_table.py it was ['qwen2.5', 'llama3.2','CoDE'].
# Using the 'llama3' version here as per the original combine_tables.py.
# Ensure MODEL_NAME_MAP_FULL aligns.
MODELS_ABBR_COMBINE = ['qwen2.5', 'llama3', 'CoDE']
MODEL_NAME_MAP_FULL_COMBINE = {'llama3': 'llama3-11b', 'qwen2.5': 'qwen25-7b'}

LLM_METHODS_COMBINE = ['zeroshot', 'zeroshot-cot', 'zeroshot-2-artifacts']
DATASETS_TO_PROCESS_COMBINE = ['d3', 'df40', 'genimage']

# Subset Naming (Same as recall_subsets_table.py, consider moving to a shared config if identical)
SUBSET_NAME_MAPPING_COMBINE = {
    "real": "real", "image_gen0": "dfif", "image_gen1": "sd1.4", "image_gen2": "sd2.1", "image_gen3": "sdxl",
    "collabdiff": "cdif", "midjourney": "midj", "stargan": "sgan1", "starganv2": "sgan2", "styleclip": "sclip",
    "whichfaceisreal": "wfir", "adm": "adm", "biggan": "bgan", "glide_copy": "glide",
    "stable_diffusion_v_1_4": "sd1.4", "stable_diffusion_v_1_5": "sd1.5", "vqdm": "vqdm", "wukong": "wk",
    "unknown_d3_subset": "Unknown D3"
}

# Method Name Mapping (for LLMs)
METHOD_DISPLAY_NAME_MAPPING_COMBINE = {
    'zeroshot': 'zeroshot',
    'zeroshot-cot': 'zeroshot-cot',
    'zeroshot-2-artifacts': r'zeroshot-s$^2$' # combine_tables uses 'zeroshot-s$^2$' without (ours)
}
# METHOD_TO_ADD_OURS_SUFFIX_COMBINE = 'zeroshot-2-artifacts' # Not used for (ours) suffix in combine_tables.py original
# For combine_tables.py, there was no diff calculation shown in the example.
# If diffs are needed, METHODS_TO_SHOW_DIFF and BASELINE_METHOD_FOR_DIFF would be needed.

# LaTeX Formatting Options
BOLD_MAX_RECALL_PER_MODEL_COMBINE = True
ZERO_PAD_RECALL_COMBINE = True
TABLE_FONT_SIZE_COMBINE = r"\small"
# DIFF_FONT_SIZE_COMBINE = r"\scriptsize" # Not used if no diffs

# CoDe specific identifiers
CODE_MODEL_ABBR_COMBINE = "CoDE"
CODE_RECALL_METHOD_KEY_COMBINE = "recall_data" # Internal key for CoDE's data
CODE_DISPLAY_METHOD_NAME_COMBINE = "trained on D3"


# --- Custom Formatting Function (Simplified as no diffs were in original combine_tables.py) ---
def format_recall_value_combine(score_val, zero_pad=True):
    """Formats recall score using helper."""
    return helpers.format_score_for_display(score_val, zero_pad=zero_pad, decimal_places=2) # Original was 2dp


# --- Main Processing Logic ---
def main():
    # --- 1. Load Dataset Mappings ---
    logger.info("Loading dataset image-to-subset mappings using helpers...")
    dataset_mappings = {
        'df40': helpers.load_df40_data_mapping(config.DF40_10K_CSV_FILE),
        'd3': helpers.load_d3_data_mapping(config.D3_DIR),
        'genimage': helpers.load_genimage_data_mapping(config.GENIMAGE_10K_CSV_FILE)
    }
    dataset_mappings = {k: v for k, v in dataset_mappings.items() if v and k in DATASETS_TO_PROCESS_COMBINE}
    if not dataset_mappings:
        logger.error("No dataset mappings loaded successfully for combine_tables.py. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded mappings for datasets: {list(dataset_mappings.keys())}")

    # --- 2. Initialize Data Structures ---
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'subsets': defaultdict(lambda: np.nan),
        'raw_counts': defaultdict(lambda: {'TP': 0, 'FN': 0})
    })))
    active_subsets_per_dataset = defaultdict(set)

    # --- 3. Process Recall Data from Response Files using Helper ---
    logger.info("Processing .jsonl response files for TP/FN counts using helpers for combine_tables.py...")
    files_processed_count = 0
    files_missing_count = 0

    for model_abbr in MODELS_ABBR_COMBINE:
        for dataset_name_key in DATASETS_TO_PROCESS_COMBINE:
            if dataset_name_key not in dataset_mappings:
                logger.warning(f"No mapping for dataset '{dataset_name_key}'. Skipping processing.")
                continue
            current_dataset_map = dataset_mappings[dataset_name_key]

            methods_to_process_for_model = []
            filename_template_str = ""

            if model_abbr == CODE_MODEL_ABBR_COMBINE:
                methods_to_process_for_model = [CODE_RECALL_METHOD_KEY_COMBINE]
                # The original combine_tables used "AI_dev" prefix for CoDE.
                filename_template_str = f"AI_dev-{dataset_name_key}-CoDE-rationales.jsonl"
            else:
                model_full_name = MODEL_NAME_MAP_FULL_COMBINE.get(model_abbr)
                if not model_full_name:
                    logger.warning(f"No full model name for LLM '{model_abbr}'. Skipping for {dataset_name_key}.")
                    continue
                methods_to_process_for_model = LLM_METHODS_COMBINE
                # The original combine_tables used "AI_dev" prefix for LLMs.
                prefix = "AI_dev" # As per original combine_tables.py
                filename_template_str = f"{prefix}-{dataset_name_key}-{model_full_name}-{{method}}-n1-rationales.jsonl"


            for method_key_internal in methods_to_process_for_model:
                actual_file_name = ""
                if model_abbr == CODE_MODEL_ABBR_COMBINE:
                    actual_file_name = filename_template_str
                else:
                    actual_file_name = filename_template_str.format(method=method_key_internal)

                rationale_file_path = config.RESPONSES_DIR / actual_file_name

                if not rationale_file_path.exists():
                    logger.info(f"Rationale file not found: {rationale_file_path}")
                    files_missing_count += 1
                    continue

                is_d3 = (dataset_name_key == 'd3') # D3 needs special handling for image keys
                logger.info(f"Processing file: {rationale_file_path} (is_d3={is_d3})")

                subset_tp_fn_counts = helpers.get_subset_recall_counts_from_rationales(
                    rationale_file_path=rationale_file_path,
                    dataset_image_to_subset_map=current_dataset_map,
                    dataset_name_for_logging=f"{model_abbr}_{method_key_internal}_{dataset_name_key}",
                    is_d3_dataset=is_d3
                )

                if subset_tp_fn_counts:
                    all_results[dataset_name_key][model_abbr][method_key_internal]['raw_counts'] = defaultdict(
                        lambda: {'TP': 0, 'FN': 0}, subset_tp_fn_counts
                    )
                    active_subsets_per_dataset[dataset_name_key].update(subset_tp_fn_counts.keys())
                    files_processed_count += 1
                else:
                    logger.warning(f"No recall counts returned by helper for {rationale_file_path}")

    logger.info(f"\nRecall File Processing Summary (combine_tables.py):")
    logger.info(f"  Files Successfully Processed for TP/FN counts: {files_processed_count}")
    logger.info(f"  Files Missing/Not Found: {files_missing_count}")

    # --- 4. Calculate Recall Scores ---
    logger.info("\nCalculating recall scores from TP/FN counts...")
    for dataset_name_key in all_results:
        for model_abbr_key in all_results[dataset_name_key]:
            for method_key_internal in all_results[dataset_name_key][model_abbr_key]:
                raw_counts_data = all_results[dataset_name_key][model_abbr_key][method_key_internal]['raw_counts']
                if not raw_counts_data:
                    logger.warning(f"No raw_counts for {dataset_name_key}/{model_abbr_key}/{method_key_internal}.")
                    continue
                for subset_name_key, counts_dict in raw_counts_data.items():
                    tp = counts_dict.get('TP', 0)
                    fn = counts_dict.get('FN', 0)
                    recall_score = helpers.get_recall(tp, fn)
                    # Original combine_tables used 2 decimal places for recall.
                    all_results[dataset_name_key][model_abbr_key][method_key_internal]['subsets'][subset_name_key] = round(recall_score * 100, 2)


    # --- 5. Generate LaTeX Tables ---
    logger.info("\nGenerating LaTeX tables for combine_tables.py...")
    final_latex_string_io = io.StringIO()
    final_latex_string_io.write("% --- Suggested LaTeX Preamble (for combine_tables.py) ---\n")
    final_latex_string_io.write("% \\usepackage{booktabs, multirow, graphicx, caption, amsmath}\n")
    final_latex_string_io.write("% \\begin{document}\n\n")

    sorted_dataset_keys = sorted(list(dataset_mappings.keys()))

    for dataset_name_key in sorted_dataset_keys:
        if dataset_name_key not in all_results or not all_results[dataset_name_key]:
            logger.info(f"Skipping table for dataset '{dataset_name_key}' (combine_tables): no results.")
            continue

        logger.info(f"Generating table for dataset: {dataset_name_key}")
        dataset_level_results_dict = all_results[dataset_name_key]
        current_active_subsets_raw_keys = sorted(list(active_subsets_per_dataset.get(dataset_name_key, set())))

        sorted_subsets_for_display_order_raw = []
        if not current_active_subsets_raw_keys:
            logger.warning(f"No active subsets for '{dataset_name_key}'. Columns might be empty.")
        else:
            for s_raw in current_active_subsets_raw_keys:
                if s_raw not in SUBSET_NAME_MAPPING_COMBINE: # Use specific mapping for this script
                    s_lower = s_raw.lower()
                    if s_lower not in SUBSET_NAME_MAPPING_COMBINE.values():
                         SUBSET_NAME_MAPPING_COMBINE[s_raw] = s_lower
                         logger.info(f"Added default display mapping for '{s_raw}' -> '{s_lower}' (combine_tables)")
                    elif SUBSET_NAME_MAPPING_COMBINE.get(s_raw) is None:
                         SUBSET_NAME_MAPPING_COMBINE[s_raw] = s_lower
            
            mapped_display_names_for_sort = {s_raw: SUBSET_NAME_MAPPING_COMBINE.get(s_raw, s_raw.lower())
                                             for s_raw in current_active_subsets_raw_keys}
            temp_sorted_raw_keys = sorted(current_active_subsets_raw_keys,
                                          key=lambda s_raw: mapped_display_names_for_sort[s_raw])
            if 'real' in temp_sorted_raw_keys:
                temp_sorted_raw_keys.remove('real')
                sorted_subsets_for_display_order_raw = ['real'] + temp_sorted_raw_keys
            else:
                sorted_subsets_for_display_order_raw = temp_sorted_raw_keys

        final_latex_string_io.write(f"% --- Table for Dataset: {dataset_name_key} (combine_tables.py) ---\n")
        final_latex_string_io.write(r"\begin{table*}[htbp]" + "\n")
        final_latex_string_io.write(r"\centering" + "\n")
        dataset_display_name_escaped = helpers.escape_latex(dataset_name_key.upper())
        # Caption from original combine_tables.py was simpler
        final_latex_string_io.write(f"\\caption{{Recall Scores for {dataset_display_name_escaped} Dataset (Combined)}}\n")
        final_latex_string_io.write(f"\\label{{tab:{dataset_name_key}_recall_combined}}\n") # Unique label
        final_latex_string_io.write(r"\setlength{\tabcolsep}{3pt}" + "\n")
        final_latex_string_io.write(TABLE_FONT_SIZE_COMBINE + "\n")

        num_subset_cols = len(sorted_subsets_for_display_order_raw)
        col_spec = 'll' + ('l' * num_subset_cols if num_subset_cols > 0 else '')
        final_latex_string_io.write(r"\begin{tabular}{" + col_spec + "}\n")
        final_latex_string_io.write(r"\toprule" + "\n")

        header_cells_list = [r"\textbf{Model}", r"\textbf{Method}"]
        for subset_raw_key_header in sorted_subsets_for_display_order_raw:
            display_col_name = SUBSET_NAME_MAPPING_COMBINE.get(subset_raw_key_header, subset_raw_key_header.lower())
            header_cells_list.append(rf"\textbf{{{helpers.escape_latex(display_col_name)}}}")
        final_latex_string_io.write(" & ".join(header_cells_list) + r" \\" + "\n")
        final_latex_string_io.write(r"\midrule" + "\n")

        models_in_table = [m for m in MODELS_ABBR_COMBINE if m in dataset_level_results_dict and dataset_level_results_dict[m]]

        for model_idx, model_abbr_render in enumerate(models_in_table):
            model_data_for_table = dataset_level_results_dict[model_abbr_render]
            
            methods_for_current_model = []
            if model_abbr_render == CODE_MODEL_ABBR_COMBINE:
                if CODE_RECALL_METHOD_KEY_COMBINE in model_data_for_table:
                    methods_for_current_model = [CODE_RECALL_METHOD_KEY_COMBINE]
            else:
                methods_for_current_model = [m for m in LLM_METHODS_COMBINE if m in model_data_for_table]

            if not methods_for_current_model:
                logger.info(f"Skipping {model_abbr_render} in {dataset_name_key} table: no method data.")
                continue

            max_recall_per_subset_for_model = defaultdict(lambda: -1.0)
            if BOLD_MAX_RECALL_PER_MODEL_COMBINE and num_subset_cols > 0 : # No model_abbr_render != CODE_MODEL_ABBR_COMBINE check here in original
                for method_k_bold in methods_for_current_model:
                    subset_scores = model_data_for_table[method_k_bold].get('subsets', {})
                    for subset_raw_k_bold in sorted_subsets_for_display_order_raw:
                        recall_val = subset_scores.get(subset_raw_k_bold, np.nan)
                        if pd.notna(recall_val) and recall_val > max_recall_per_subset_for_model[subset_raw_k_bold]:
                            max_recall_per_subset_for_model[subset_raw_k_bold] = recall_val
            
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
                if model_abbr_render == CODE_MODEL_ABBR_COMBINE:
                    method_display_name = helpers.escape_latex(CODE_DISPLAY_METHOD_NAME_COMBINE)
                else:
                    method_display_name = helpers.escape_latex(METHOD_DISPLAY_NAME_MAPPING_COMBINE.get(method_key_render, method_key_render))
                    # Original combine_tables.py used a suffix like "(ours)" for zeroshot-2-artifacts
                    # It was METHOD_TO_ADD_OURS_SUFFIX = 'zeroshot-2-artifacts'
                    # Diff font was DIFF_FONT_SIZE = r"\scriptsize"
                    # This was added: method_display_str += f"\\ {DIFF_FONT_SIZE}{{(ours)}}"
                    # Keeping this specific logic for combine_tables.py:
                    if method_key_render == 'zeroshot-2-artifacts': # Specific check
                         scriptsize_str = r'\scriptsize'
                         method_display_name += f"\\ {scriptsize_str}{{(ours)}}" # Use variable for font size

                row_cells_list.append(method_display_name)

                current_method_subset_scores = model_data_for_table[method_key_render].get('subsets', {})
                for subset_raw_key_cell in sorted_subsets_for_display_order_raw:
                    recall_value = current_method_subset_scores.get(subset_raw_key_cell, np.nan)
                    formatted_cell_value = format_recall_value_combine(
                        recall_value, zero_pad=ZERO_PAD_RECALL_COMBINE
                    )
                    current_max_recall = max_recall_per_subset_for_model.get(subset_raw_key_cell, -1.0)
                    # Original combine_tables.py bolded CoDE as well if it was max
                    if BOLD_MAX_RECALL_PER_MODEL_COMBINE and \
                       pd.notna(recall_value) and np.isclose(recall_value, current_max_recall):
                        formatted_cell_value = f"\\textbf{{{formatted_cell_value}}}"
                    row_cells_list.append(formatted_cell_value)
                
                final_latex_string_io.write(" & ".join(row_cells_list) + r" \\" + "\n")

            if model_idx < len(models_in_table) - 1:
                final_latex_string_io.write(r"\midrule" + "\n")

        final_latex_string_io.write(r"\bottomrule" + "\n")
        final_latex_string_io.write(r"\end{tabular}" + "\n")
        final_latex_string_io.write(r"\end{table*}" + "\n\n")

    final_latex_string_io.write("% --- End of Generated Tables (combine_tables.py) ---\n")
    final_latex_string_io.write("% \\end{document}\n")

    final_latex_output = final_latex_string_io.getvalue()
    final_latex_string_io.close()

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_tex_filename = "combined_recall_results_tables_refactored.tex" # New name
    output_tex_filepath = config.TABLES_DIR / output_tex_filename

    try:
        with open(output_tex_filepath, 'w', encoding='utf-8') as f:
            f.write(final_latex_output)
        logger.info(f"LaTeX table code for combine_tables.py saved to: {output_tex_filepath}")
    except IOError as e:
        logger.error(f"Error saving LaTeX for combine_tables.py: {e}", exc_info=True)

    logger.info("\nNotes for Refactored combine_tables.py:")
    logger.info("- Uses helpers for data loading, subset recall TP/FN counting, and LaTeX formatting.")
    logger.info("- Formatting (bolding, (ours) suffix) aims to replicate original combine_tables.py behavior.")

if __name__ == "__main__":
    main()