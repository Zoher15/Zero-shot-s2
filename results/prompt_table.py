import os
import sys
from pathlib import Path
import logging
import io

import pandas as pd
import numpy as np
# Removed re and json as they are likely not needed directly after refactoring

# --- Project Setup ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import config
from utils import helpers

# --- Logger Setup ---
helpers.setup_global_logger(config.RESULTS_PROMPT_TABLE_LOG_FILE)
# Get a logger instance for this specific module.
logger = logging.getLogger(__name__)

# --- Seed for reproducibility (from original script) ---
np.random.seed(0)

# --- Configuration Constants (specific to this script's table generation) ---
# Consider moving to config.py if these become more globally used or complex
TARGET_MODEL_NAME = "qwen25-7b" # Example, ensure this is set as intended

# --- Category Definition Function (from original, kept local as it's specific to table structure) ---
def get_method_category(method_key, table_type=None):
    """Determines the category of a method based on its key for the specified table type."""
    # Logic from original script (remains unchanged)
    if table_type == 2:
        if method_key == "zeroshot": return "None"
        elif method_key in ["zeroshot-q4", "zeroshot-q5"]: return "User"
        elif method_key in ["zeroshot-cot", "zeroshot-2-artifacts"]: return "Assistant"
        else: return "Unknown T2 Category"
    if method_key == "zeroshot-cot": return "Open-ended"
    elif "artifacts" in method_key: return "Task-aligned"
    elif method_key in ["zeroshot-visualize", "zeroshot-examine", "zeroshot-pixel",
                        "zeroshot-zoom", "zeroshot-flaws", "zeroshot-texture", "zeroshot-style"]:
        return "Open-ended"
    else: return "Open-ended"

# --- Configuration for Table 1 (General Prompts) ---
METHOD_PROMPT_MAP_TABLE1 = {
    "zeroshot": "", "zeroshot-cot": "Let's think step by step",
    "zeroshot-visualize": "Let's visualize", "zeroshot-examine": "Let's examine",
    "zeroshot-pixel": "Let's examine pixel by pixel", "zeroshot-zoom": "Let's zoom in",
    "zeroshot-flaws": "Let's examine the flaws", "zeroshot-texture": "Let's examine the textures",
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
    "zeroshot-q4": "Is this image real or AI-generated? Let's think step by step",
    "zeroshot-q5": "Is this image real or AI-generated? Let's examine the style and the synthesis artifacts",
}
ORDERED_METHOD_KEYS_TABLE1 = [
    "zeroshot-visualize", "zeroshot-examine", "zeroshot-pixel", "zeroshot-zoom",
    "zeroshot-flaws", "zeroshot-texture", "zeroshot-style", "zeroshot-cot",
    "zeroshot-artifacts", "zeroshot-3-artifacts", "zeroshot-4-artifacts",
    "zeroshot-5-artifacts", "zeroshot-6-artifacts", "zeroshot-7-artifacts",
    "zeroshot-8-artifacts", "zeroshot-9-artifacts", "zeroshot-2-artifacts"
]

# --- Configuration for Table 2 (Specific LaTeX-Formatted Prompts) ---
METHOD_PROMPT_MAP_TABLE2 = {
    "zeroshot": {"user_content": r"{\fontseries{b}\selectfont User:} [Image] Is this image real or AI-generated?", "assistant_prefix": r"{\fontseries{b}\selectfont Assistant:}", "assistant_figure_input": None, "assistant_suffix": ""},
    "zeroshot-q4": {"user_content": r"{\fontseries{b}\selectfont User:} [Image] Is this image real or AI-generated? Let's think step by step", "assistant_prefix": r"{\fontseries{b}\selectfont Assistant:}", "assistant_figure_input": None, "assistant_suffix": ""},
    "zeroshot-q5": {"user_content": r"{\fontseries{b}\selectfont User:} [Image] Is this image real or AI-generated? Let's examine the style and the synthesis artifacts", "assistant_prefix": r"{\fontseries{b}\selectfont Assistant:}", "assistant_figure_input": None, "assistant_suffix": ""},
    "zeroshot-cot": {"user_content": r"{\fontseries{b}\selectfont User:} [Image] Is this image real or AI-generated?", "assistant_prefix": r"{\fontseries{b}\selectfont Assistant:}", "assistant_figure_input": r"Let's think step by step", "assistant_suffix": ""},
    "zeroshot-2-artifacts": {"user_content": r"{\fontseries{b}\selectfont User:} [Image] Is this image real or AI-generated?", "assistant_prefix": r"{\fontseries{b}\selectfont Assistant:}", "assistant_figure_input": r"Let's examine the style and the synthesis artifacts", "assistant_suffix": ""},
}
ORDERED_METHOD_KEYS_TABLE2 = ["zeroshot", "zeroshot-q4", "zeroshot-q5", "zeroshot-cot", "zeroshot-2-artifacts"]

# Shared settings (Consider moving to config.py or a plotting/table config if widely used)
ALLOWED_DATASETS = ["d32k", "df402k", "genimage2k"] # Used by this script
N_VAL_TABLE = "1"
WAIT_VAL_TABLE = "0" # If your eval scripts save with -wait0, otherwise this might not be needed for filename
BOLD_MAX_SCORE_PER_DATASET_COLUMN = True
TABLE_FONT_SIZE = r"\scriptsize"
DATASET_DISPLAY_MAP_TABLE = {
    "d32k": "D3 (2k)", "df402k": "DF40 (2k)", "genimage2k": "GenImage (2k)"
}

# Model Family and Size Mapping (will be populated using helpers.parse_model_name)
MODEL_FAMILY_MAP = {}
MODEL_SIZE_MAP = {}

# --- Data Loading (Refactored to use helpers.load_scores_csv_to_dataframe) ---
def load_data_and_calculate_scores(
    model_to_load: str,
    current_allowed_datasets: list,
    method_keys_to_load: list,
    n_val_param: str,
    # wait_val_param: str, # wait_val might not be in filenames from eval script
    all_method_maps_for_validation: list
):
    logger.info(f"Loading Scores from CSV files in: {config.SCORES_DIR} for model: {model_to_load}")
    all_data_list = []
    stats = {"csv_read": 0, "missing_csvs": 0, "errors_reading_csv": 0, "files_checked": 0}

    valid_keys_for_loading = set()
    for m_map in all_method_maps_for_validation:
        valid_keys_for_loading.update(m_map.keys())

    actual_keys_to_process_loading = [key for key in method_keys_to_load if key in valid_keys_for_loading]
    if len(actual_keys_to_process_loading) != len(method_keys_to_load):
        skipped_keys = set(method_keys_to_load) - set(actual_keys_to_process_loading)
        logger.info(f"The following method keys specified for loading were not found in any METHOD_PROMPT_MAP and will be skipped: {sorted(list(skipped_keys))}")

    for dataset_name_table in current_allowed_datasets:
        for method_key in actual_keys_to_process_loading:
            stats["files_checked"] += 1
            score_value = None # Initialize score_value

            # Construct filename based on how evaluate_AI_*.py scripts save them (via helpers.save_evaluation_outputs)
            # The helper saves as: f"{model_prefix}-{dataset_name}-{model_string}-{mode_type_str}-n{num_sequences_val}-scores.csv"
            prefix = "AI_llama" if "llama" in model_to_load.lower() else "AI_qwen"
            # Assuming WAIT_VAL_TABLE '0' means the -wait0 part is omitted by eval script if wait is 0
            # If eval scripts *always* include -wait0, then add it back here:
            # fname_csv = f"{prefix}-{dataset_name_table}-{model_to_load}-{method_key}-n{n_val_param}-wait{WAIT_VAL_TABLE}-scores.csv"
            fname_csv = f"{prefix}-{dataset_name_table}-{model_to_load}-{method_key}-n{n_val_param}-scores.csv"
            fpath_csv = config.SCORES_DIR / fname_csv

            df_score_from_file = helpers.load_scores_csv_to_dataframe(fpath_csv)

            if not df_score_from_file.empty:
                try:
                    # The index in CSV is like 'zeroshot-n1'
                    target_csv_index = f"{method_key}-n{n_val_param}"
                    if target_csv_index in df_score_from_file.index:
                        # The score is in a column, likely 'macro_f1' or the first column
                        score_col_name = 'macro_f1' if 'macro_f1' in df_score_from_file.columns else df_score_from_file.columns[0]
                        raw_score = df_score_from_file.loc[target_csv_index, score_col_name]
                        
                        score_value = round(float(raw_score) * 100, 1) # Scores in CSV are 0-1, table wants 0-100
                        stats["csv_read"] += 1
                        logger.info(f"  CSV READ: {model_to_load}/{dataset_name_table}/{method_key}. Score: {score_value:.1f}")
                    else:
                        logger.warning(f"Index '{target_csv_index}' not found in CSV {fpath_csv}. Available: {df_score_from_file.index.tolist()}")
                        stats["errors_reading_csv"] += 1
                except KeyError: # Handles if target_csv_index is not found by .loc
                    logger.warning(f"Index '{target_csv_index}' caused KeyError in CSV {fpath_csv}.")
                    stats["errors_reading_csv"] += 1
                except Exception as e:
                    logger.error(f"Error processing data from CSV {fpath_csv}: {e}", exc_info=True)
                    stats["errors_reading_csv"] += 1
            else:
                # helpers.load_scores_csv_to_dataframe already logs if file not found.
                # This part handles cases where the file might exist but is empty or unreadable by the helper.
                if fpath_csv.exists():
                     logger.warning(f"CSV file {fpath_csv} exists but was empty or unreadable by helper.")
                # else: # Already logged by helper
                #    logger.info(f"CSV score file NOT FOUND: {fpath_csv}")
                stats["missing_csvs"] += 1
            
            table_type_for_category = 2 if method_key in METHOD_PROMPT_MAP_TABLE2 else 1
            category_for_df = get_method_category(method_key, table_type=table_type_for_category)

            all_data_list.append({
                'llm': model_to_load,
                'dataset': dataset_name_table,
                'method': method_key,
                'category': category_for_df,
                'score': score_value, # This will be None if not found/error
                'n_samples': 0 # n_samples not directly available from these summary CSVs
            })

    logger.info(f"Finished loading scores. Files Checked: {stats['files_checked']}, CSVs Read: {stats['csv_read']}, Missing: {stats['missing_csvs']}, Errors: {stats['errors_reading_csv']}")
    if not all_data_list:
        return pd.DataFrame()
    
    df_return = pd.DataFrame(all_data_list)
    # Populate MODEL_FAMILY_MAP and MODEL_SIZE_MAP dynamically
    unique_llms = df_return['llm'].unique()
    for llm_name_str in unique_llms:
        if llm_name_str not in MODEL_FAMILY_MAP: # Parse only if not already parsed
            fam, size = helpers.parse_model_name(llm_name_str)
            MODEL_FAMILY_MAP[llm_name_str] = fam
            MODEL_SIZE_MAP[llm_name_str] = size

    df_return['model_family'] = df_return['llm'].map(MODEL_FAMILY_MAP)
    df_return['model_size'] = df_return['llm'].map(MODEL_SIZE_MAP)
    return df_return

# --- Main LaTeX Table Generation Logic (Adapted from original, uses helpers now) ---
def generate_prompt_focused_latex_table(
    scores_df: pd.DataFrame,
    target_model_name_str: str,
    current_method_prompt_map: dict,
    current_ordered_method_keys: list,
    current_allowed_datasets: list,
    current_dataset_display_map: dict,
    table_caption_str: str,
    table_label_str: str,
    table_headers_list: list,
    wrap_general_prompt_with_inlineinput: bool = True,
    use_minipage_for_structured_prompts: bool = False
):
    if scores_df.empty:
        logger.warning(f"No data to generate table ({table_label_str}). Scores DataFrame is empty.")
        return f"% No data to generate table ({table_label_str}). Scores DataFrame is empty.\n"
    
    model_scores_df = scores_df[scores_df['llm'] == target_model_name_str].copy()
    if model_scores_df.empty:
        logger.warning(f"No data for model {target_model_name_str} in scores DataFrame for table {table_label_str}.")
        return f"% No data for model {target_model_name_str} for table {table_label_str}.\n"

    final_latex_io = io.StringIO()
    final_latex_io.write(f"% --- Table: {table_label_str} ---\n")
    final_latex_io.write(r"\begin{table*}[htbp!]" + "\n")
    final_latex_io.write(r"\centering" + "\n")
    final_latex_io.write(f"\\caption{{{table_caption_str}}}\n") # Caption is already escaped if needed by caller
    final_latex_io.write(f"\\label{{{table_label_str}}}\n")
    final_latex_io.write(TABLE_FONT_SIZE + "\n") # Global TABLE_FONT_SIZE

    current_table_type = 2 if use_minipage_for_structured_prompts else 1
    category_col_width = "0.07\\textwidth" if current_table_type == 2 else "0.09\\textwidth"
    prompt_col_width = "0.53\\textwidth" if current_table_type == 2 else "0.515\\textwidth"
    prompt_col_type = 'm' if current_table_type == 2 else 'p'
    
    col_spec = f"m{{{category_col_width}}}{prompt_col_type}{{{prompt_col_width}}}" + ('c' * len(current_allowed_datasets))
    
    final_latex_io.write(r"\begin{tabular}{" + col_spec + "}\n")
    final_latex_io.write(r"\toprule" + "\n")
    
    latex_header_cells = [rf"\textbf{{{header}}}" for header in table_headers_list] + \
                         [rf"\textbf{{{helpers.escape_latex(current_dataset_display_map.get(ds_key, ds_key.upper()))}}}"
                          for ds_key in current_allowed_datasets]
    final_latex_io.write(" & ".join(latex_header_cells) + r" \\" + "\n")
    final_latex_io.write(r"\midrule" + "\n")
    
    max_scores_per_dataset_col = {}
    if BOLD_MAX_SCORE_PER_DATASET_COLUMN:
        for ds_key_bold in current_allowed_datasets:
            relevant_scores_for_max = model_scores_df[
                model_scores_df['method'].isin(current_ordered_method_keys) &
                (model_scores_df['dataset'] == ds_key_bold)
            ]['score']
            dataset_scores_numeric = pd.to_numeric(relevant_scores_for_max, errors='coerce')
            if not dataset_scores_numeric.empty and dataset_scores_numeric.notna().any():
                max_scores_per_dataset_col[ds_key_bold] = dataset_scores_numeric.max()
            else:
                max_scores_per_dataset_col[ds_key_bold] = -1 # Default if no valid scores

    row_category_details_list = []
    if current_ordered_method_keys:
        method_categories_plain_text = [get_method_category(mk, table_type=current_table_type) for mk in current_ordered_method_keys]
        idx = 0
        num_total_methods = len(current_ordered_method_keys)
        while idx < num_total_methods:
            current_cat_text = method_categories_plain_text[idx]
            count = method_categories_plain_text.count(current_cat_text) # Simplified count for contiguous blocks
            # This simple count assumes categories are grouped in ORDERED_METHOD_KEYS.
            # For true grouping if non-contiguous, a more complex loop is needed as in original.
            # For now, assuming ORDERED_METHOD_KEYS is structured by category.
            # Reverting to original more robust grouping logic:
            count_in_group = 0
            temp_idx = idx
            while temp_idx < num_total_methods and method_categories_plain_text[temp_idx] == current_cat_text:
                count_in_group +=1
                temp_idx +=1

            for k_in_group_idx in range(count_in_group):
                row_details = {
                    'method_key': current_ordered_method_keys[idx + k_in_group_idx],
                    'category_name_plain': current_cat_text,
                    'is_first_in_group': (k_in_group_idx == 0),
                    'group_span': count_in_group if (k_in_group_idx == 0) else 0,
                    'is_last_in_group': (k_in_group_idx == count_in_group - 1)
                }
                row_category_details_list.append(row_details)
            idx += count_in_group

    num_rows_to_generate = len(row_category_details_list)
    for i, details_dict in enumerate(row_category_details_list):
        method_key_current = details_dict['method_key']
        if method_key_current not in current_method_prompt_map:
            logger.warning(f"Key '{method_key_current}' in ordered list for table '{table_label_str}' not in its map. Skipping.")
            continue
        
        prompt_data_current = current_method_prompt_map.get(method_key_current)
        row_cells_list_current = []
        
        if details_dict['is_first_in_group']:
            multirow_cat_text = helpers.escape_latex(details_dict['category_name_plain'])
            row_cells_list_current.append(rf"\multirow{{{details_dict['group_span']}}}{{*}}{{{multirow_cat_text}}}")
        else:
            row_cells_list_current.append("")

        final_prompt_cell_str = ""
        if use_minipage_for_structured_prompts: # Table 2 logic
            if not isinstance(prompt_data_current, dict):
                logger.error(f"({table_label_str}): Prompt data for key '{method_key_current}' is not a dict. Skipping prompt.")
                final_prompt_cell_str = f"Error: Bad prompt data for {method_key_current}"
            else:
                user_part = prompt_data_current.get("user_content", "")
                assistant_prefix = prompt_data_current.get("assistant_prefix", "")
                figure_input = prompt_data_current.get("assistant_figure_input")
                assistant_suffix = prompt_data_current.get("assistant_suffix", "")
                assistant_combined = assistant_prefix.strip()
                if figure_input: assistant_combined += rf" \figureinput{{{figure_input.strip()}}}"
                if assistant_suffix.strip(): assistant_combined += " " + assistant_suffix.strip()
                assembled_text = rf"\inlineinputt{{{user_part.strip()}}} \\ \inlineoutputt{{{assistant_combined.strip()}}}"
                final_prompt_cell_str = rf"\begin{{minipage}}[c]{{\linewidth}}{{{assembled_text}}}\end{{minipage}}"
        else: # Table 1 logic
            base_text_t1 = prompt_data_current if prompt_data_current else method_key_current
            escaped_text_t1 = helpers.escape_latex(base_text_t1)
            if wrap_general_prompt_with_inlineinput:
                final_prompt_cell_str = rf"\inlineinput{{{escaped_text_t1}}}"
            else:
                final_prompt_cell_str = escaped_text_t1
        row_cells_list_current.append(final_prompt_cell_str)

        for ds_key_cell in current_allowed_datasets:
            score_entry_df = model_scores_df[
                (model_scores_df['method'] == method_key_current) & (model_scores_df['dataset'] == ds_key_cell)
            ]
            score_val_cell = score_entry_df['score'].iloc[0] if not score_entry_df.empty and pd.notna(score_entry_df['score'].iloc[0]) else np.nan
            
            # Use helpers.format_score_for_display
            formatted_score_str = helpers.format_score_for_display(score_val_cell, zero_pad=config.ZERO_PAD_SCORE_TABLES, decimal_places=1) # Assuming ZERO_PAD_SCORE_TABLES is in config

            if BOLD_MAX_SCORE_PER_DATASET_COLUMN and pd.notna(score_val_cell) and \
               ds_key_cell in max_scores_per_dataset_col and \
               np.isclose(score_val_cell, max_scores_per_dataset_col[ds_key_cell]) and formatted_score_str != '-':
                formatted_score_str = rf"\textbf{{{formatted_score_str}}}"
            row_cells_list_current.append(formatted_score_str)
            
        final_latex_io.write(" & ".join(row_cells_list_current) + r" \\" + "\n")
        
        if details_dict['is_last_in_group'] and i < num_rows_to_generate - 1:
            final_latex_io.write(r"\hline" + "\n")
        
    final_latex_io.write(r"\bottomrule" + "\n")
    final_latex_io.write(r"\end{tabular}" + "\n")
    final_latex_io.write(r"\end{table*}" + "\n\n% --- End of Table ---\n")
    return final_latex_io.getvalue()

# --- Key Validation Function (from original, seems fine) ---
def validate_configuration(ordered_keys_list, method_map_dict, table_name_str):
    logger.info(f"Validating Configuration for {table_name_str}")
    is_valid_config = True
    map_keys_set_val = set(method_map_dict.keys())
    processed_ordered_keys_list = []
    seen_in_ordered_set = set()

    for key_val in ordered_keys_list:
        if key_val in seen_in_ordered_set:
            logger.error(f"({table_name_str}): Duplicate key '{key_val}' in ORDERED_METHOD_KEYS. Ensure uniqueness.")
            is_valid_config = False
        else:
            seen_in_ordered_set.add(key_val)
            processed_ordered_keys_list.append(key_val)

    for key_val in processed_ordered_keys_list:
        if key_val not in map_keys_set_val:
            logger.error(f"({table_name_str}): Key '{key_val}' in ORDERED_METHOD_KEYS not in METHOD_PROMPT_MAP.")
            is_valid_config = False

    unused_map_keys_set = map_keys_set_val - seen_in_ordered_set
    if unused_map_keys_set:
        logger.info(f"({table_name_str}): Keys in METHOD_PROMPT_MAP but not in ORDERED_METHOD_KEYS (will not be in table): {sorted(list(unused_map_keys_set))}")
    
    if not ordered_keys_list:
        logger.warning(f"({table_name_str}): ORDERED_METHOD_KEYS list is empty. No rows will be generated.")
    return is_valid_config

# --- Main Execution ---
if __name__ == "__main__":
    logger.info(f"--- Starting Prompt Table Script for Model: {TARGET_MODEL_NAME} ---")
    if not config.SCORES_DIR.is_dir():
        logger.error(f"SCORES_DIR '{config.SCORES_DIR}' not found. Exiting.")
        sys.exit(1)
    
    # Add a global config for ZERO_PAD_SCORE_TABLES if using helpers.format_score_for_display
    # For now, assuming it's True as per original script's ZERO_PAD_SCORE
    if not hasattr(config, 'ZERO_PAD_SCORE_TABLES'):
        logger.warning("config.ZERO_PAD_SCORE_TABLES not found, defaulting to True for score padding.")
        config.ZERO_PAD_SCORE_TABLES = True


    valid_config_t1 = validate_configuration(ORDERED_METHOD_KEYS_TABLE1, METHOD_PROMPT_MAP_TABLE1, "Table 1 (General Prompts)")
    valid_config_t2 = validate_configuration(ORDERED_METHOD_KEYS_TABLE2, METHOD_PROMPT_MAP_TABLE2, "Table 2 (LaTeX Prompts)")

    if not (valid_config_t1 and valid_config_t2):
        logger.error("Exiting due to configuration validation errors.")
        sys.exit(1)

    all_required_method_keys = list(set(ORDERED_METHOD_KEYS_TABLE1) | set(ORDERED_METHOD_KEYS_TABLE2))
    if not all_required_method_keys:
        logger.error("No methods specified in any table configuration. Exiting.")
        sys.exit(1)
        
    logger.info(f"Will attempt to load data for unique method keys: {sorted(all_required_method_keys)}")

    main_scores_df = load_data_and_calculate_scores(
        TARGET_MODEL_NAME, ALLOWED_DATASETS,
        all_required_method_keys, N_VAL_TABLE, # Removed WAIT_VAL_TABLE if not in filenames
        all_method_maps_for_validation=[METHOD_PROMPT_MAP_TABLE1, METHOD_PROMPT_MAP_TABLE2]
    )

    if main_scores_df.empty and all_required_method_keys:
        logger.error("Exiting: No data loaded or calculated overall. Check CSV score files and paths.")
        sys.exit(1)
    
    latex_preamble_io = io.StringIO()
    latex_preamble_io.write("% --- Suggested LaTeX Preamble (ensure these are in your main .tex file) ---\n")
    # ... (preamble text from original, unchanged)
    latex_preamble_io.write("% \\begin{document}\n\n")
    full_latex_output_str = latex_preamble_io.getvalue()

    # --- Generate and Save First Table ---
    if ORDERED_METHOD_KEYS_TABLE1:
        logger.info("Generating First Table (General Prompts)")
        caption_t1 = f"Effect of phrase types on Macro F1 scores (\%) for {helpers.escape_latex(TARGET_MODEL_NAME)}."
        label_t1 = f"tab:prompt_performance_{TARGET_MODEL_NAME.replace('-', '_')}_general_nested_cat"
        headers_t1 = ["Category", "Prompt Phrase"]
        latex_output_t1 = generate_prompt_focused_latex_table(
            main_scores_df, TARGET_MODEL_NAME, METHOD_PROMPT_MAP_TABLE1, ORDERED_METHOD_KEYS_TABLE1,
            ALLOWED_DATASETS, DATASET_DISPLAY_MAP_TABLE, caption_t1, label_t1, headers_t1,
            wrap_general_prompt_with_inlineinput=True,
            use_minipage_for_structured_prompts=False
        )
        full_latex_output_str += latex_output_t1
        # logger.info("\n" + "="*25 + " Generated LaTeX Code (First Table) " + "="*25) # For debug
        # logger.info(latex_output_t1)
        # logger.info("="*70)
    else:
        logger.info("Skipping generation of Table 1: ORDERED_METHOD_KEYS_TABLE1 is empty.")

    # --- Generate and Save Second Table ---
    if ORDERED_METHOD_KEYS_TABLE2:
        logger.info("Generating Second Table (Structured LaTeX Prompts)")
        caption_t2 = f"Effect of phrase placement on Macro F1 scores (\%) for {helpers.escape_latex(TARGET_MODEL_NAME)}."
        label_t2 = f"tab:prompt_performance_{TARGET_MODEL_NAME.replace('-', '_')}_structured_minipage_tt_nested_cat"
        headers_t2 = ["Placement", "Full Prompt"]
        latex_output_t2 = generate_prompt_focused_latex_table(
            main_scores_df, TARGET_MODEL_NAME, METHOD_PROMPT_MAP_TABLE2, ORDERED_METHOD_KEYS_TABLE2,
            ALLOWED_DATASETS, DATASET_DISPLAY_MAP_TABLE, caption_t2, label_t2, headers_t2,
            wrap_general_prompt_with_inlineinput=False,
            use_minipage_for_structured_prompts=True
        )
        full_latex_output_str += "\n\n" + latex_output_t2
        # logger.info("\n" + "="*25 + " Generated LaTeX Code (Second Table) " + "="*25) # For debug
        # logger.info(latex_output_t2)
        # logger.info("="*70)
    else:
        logger.info("Skipping generation of Table 2: ORDERED_METHOD_KEYS_TABLE2 is empty.")

    full_latex_output_str += "\n% \\end{document}\n"

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    combined_output_filename = f"{TARGET_MODEL_NAME.replace('.', '_')}_combined_prompt_tables_refactored.tex"
    combined_output_filepath = config.TABLES_DIR / combined_output_filename

    try:
        with open(combined_output_filepath, 'w', encoding='utf-8') as f:
            f.write(full_latex_output_str)
        logger.info(f"Combined LaTeX code for prompt tables saved to: {combined_output_filepath}")
    except IOError as e:
        logger.error(f"Error saving combined LaTeX file: {e}", exc_info=True)

    logger.info("--- Prompt Table Script Finished ---")
    # Original notes can be logged as well if desired.