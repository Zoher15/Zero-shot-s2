import os
import pandas as pd
import json
import io
import re
import numpy as np
import sys
from pathlib import Path # <--- ADD

# Assuming config.py is in the project root (parent of 'results')
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
import config # <--- ADD

# --- Seed for reproducibility ---
np.random.seed(0)

# --- Configuration ---
SCORES_DIR = config.SCORES_DIR # <--- CHANGED
CI_CACHE_DIR_TABLE = config.PROMPT_TABLE_CACHE_DIR # <--- CHANGED
TARGET_MODEL_NAME = "qwen25-7b"

# --- Category Definition Function (Updated based on user's final table structure) ---
def get_method_category(method_key, table_type=None):
    """Determines the category of a method based on its key for the specified table type."""
    if table_type == 2: # Categories for Table 2 - as per user's final example image
        if method_key == "zeroshot":
            return "None" 
        elif method_key in ["zeroshot-q4", "zeroshot-q5"]:
            return "User" 
        elif method_key in ["zeroshot-cot", "zeroshot-2-artifacts"]:
            return "Assistant" 
        else: 
            return "Unknown T2 Category" 

    # Categories for Table 1 - based on user's final Table 1 example
    if method_key == "zeroshot-cot": 
        return "Open-ended"
    elif "artifacts" in method_key: # This will catch "zeroshot-2-artifacts" and others like "zeroshot-artifacts"
        return "Task-aligned"
    # Add other specific Table 1 categories if needed, otherwise default to Open-ended
    elif method_key in ["zeroshot-visualize", "zeroshot-examine", "zeroshot-pixel", 
                        "zeroshot-zoom", "zeroshot-flaws", "zeroshot-texture", "zeroshot-style"]:
        return "Open-ended"
    else: # Fallback for any other Table 1 methods not explicitly categorized
        return "Open-ended"


# --- Configuration for Table 1 (General Prompts) ---
METHOD_PROMPT_MAP_TABLE1 = {
    "zeroshot": "", 
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
    "zeroshot-q4": "Is this image real or AI-generated? Let's think step by step", 
    "zeroshot-q5": "Is this image real or AI-generated? Let's examine the style and the synthesis artifacts", 
}

# Order for Table 1, matching user's final example structure
ORDERED_METHOD_KEYS_TABLE1 = [
    # Open-ended (8 prompts in user example)
    "zeroshot-visualize", 
    "zeroshot-examine", 
    "zeroshot-pixel", 
    "zeroshot-zoom",
    "zeroshot-flaws", 
    "zeroshot-texture", 
    "zeroshot-style", 
    "zeroshot-cot",      # "Let's think step by step"
    # Task-aligned (9 prompts in user example)
    "zeroshot-artifacts",
    "zeroshot-3-artifacts",
    "zeroshot-4-artifacts",
    "zeroshot-5-artifacts",
    "zeroshot-6-artifacts",
    "zeroshot-7-artifacts",
    "zeroshot-8-artifacts",
    "zeroshot-9-artifacts",
    "zeroshot-2-artifacts" 
]


# --- Configuration for Table 2 (Specific LaTeX-Formatted Prompts - New Structure) ---
METHOD_PROMPT_MAP_TABLE2 = {
    "zeroshot": {
        "user_content": r"{\fontseries{b}\selectfont User:} [Image] Is this image real or AI-generated?",
        "assistant_prefix": r"{\fontseries{b}\selectfont Assistant:}",
        "assistant_figure_input": None,
        "assistant_suffix": "" 
    },
    "zeroshot-q4": {
        "user_content": r"{\fontseries{b}\selectfont User:} [Image] Is this image real or AI-generated? Let's think step by step",
        "assistant_prefix": r"{\fontseries{b}\selectfont Assistant:}",
        "assistant_figure_input": None,
        "assistant_suffix": "" 
    },
    "zeroshot-q5": {
        "user_content": r"{\fontseries{b}\selectfont User:} [Image] Is this image real or AI-generated? Let's examine the style and the synthesis artifacts",
        "assistant_prefix": r"{\fontseries{b}\selectfont Assistant:}",
        "assistant_figure_input": None,
        "assistant_suffix": "" 
    },
    "zeroshot-cot": {
        "user_content": r"{\fontseries{b}\selectfont User:} [Image] Is this image real or AI-generated?",
        "assistant_prefix": r"{\fontseries{b}\selectfont Assistant:}",
        "assistant_figure_input": r"Let's think step by step", 
        "assistant_suffix": "" 
    },
    "zeroshot-2-artifacts": { 
        "user_content": r"{\fontseries{b}\selectfont User:} [Image] Is this image real or AI-generated?",
        "assistant_prefix": r"{\fontseries{b}\selectfont Assistant:}",
        "assistant_figure_input": r"Let's examine the style and the synthesis artifacts", 
        "assistant_suffix": "" 
    },
}
# Order for the second table
ORDERED_METHOD_KEYS_TABLE2 = ["zeroshot", "zeroshot-q4", "zeroshot-q5", "zeroshot-cot", "zeroshot-2-artifacts"]


ALLOWED_DATASETS = ["d32k", "df402k", "genimage2k"]
N_VAL_TABLE = "1" # Used for filename construction
WAIT_VAL_TABLE = "0" # Used for filename construction

# --- LaTeX Table Formatting Options (Common) ---
BOLD_MAX_SCORE_PER_DATASET_COLUMN = True
ZERO_PAD_SCORE = True
TABLE_FONT_SIZE = r"\scriptsize" 
DATASET_DISPLAY_MAP_TABLE = {
    "d32k": "D3 (2k)",
    "df402k": "DF40 (2k)",
    "genimage2k": "GenImage (2k)"
}

# --- Model Parsing (Simplified) ---
MODEL_FAMILY_MAP = {}
MODEL_SIZE_MAP = {}
def _parse_model_name(model_name_config):
    match = re.match(r'([a-zA-Z]+)(\d+(\.\d+)?)?-(\d+[bB])', model_name_config)
    if match:
        base_name, version, _, size_str = match.groups()
        if base_name == 'qwen' and version == '25':
            family_name = 'qwen2.5'
        elif version:
            family_name = f"{base_name}{version}"
        else:
            family_name = base_name
        return family_name, size_str.lower()
    print(f"Warning: Could not parse family/size from LLM name: {model_name_config}", file=sys.stderr)
    return "unknown_llm_family", "unknown_llm_size"

FAMILY_NAME, SIZE = _parse_model_name(TARGET_MODEL_NAME)
MODEL_FAMILY_MAP[TARGET_MODEL_NAME] = FAMILY_NAME
MODEL_SIZE_MAP[TARGET_MODEL_NAME] = SIZE

# --- Helper Functions ---
def escape_latex(text):
    if text is None: return ''
    text_str = str(text) 
    if not text_str: return '' 
    # Order matters: escape backslash first
    text_str = text_str.replace('\\', r'\textbackslash{}') 
    replacements = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', 
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', 
        '^': r'\textasciicircum{}'
    }
    for char, escaped_char in replacements.items():
        text_str = text_str.replace(char, escaped_char)
    return text_str


def _format_number_for_score(num_val):
    if pd.isna(num_val) or num_val is None: return "-" 
    formatted_val = f"{float(num_val):.1f}"
    if ZERO_PAD_SCORE and 0 <= float(num_val) < 10: formatted_val = f"0{formatted_val}"
    return formatted_val

def format_score_for_table(score):
    if pd.isna(score) or score is None: return '-'
    return _format_number_for_score(score)

# --- Data Loading (Reads scores from CSV - CACHING DISABLED) ---
def load_data_and_calculate_scores(scores_dir_path, model_to_load, allowed_datasets,
                                   method_keys_to_load, n_val_param, wait_val_param, all_method_maps):
    print(f"\n--- Loading Scores from CSV files in: {scores_dir_path} (Caching Disabled) ---")
    all_data = []
    stats = {"csv_read": 0, "missing_csvs": 0, "errors_reading_csv": 0, "files_checked": 0}
    
    valid_keys_for_loading = set()
    for m_map in all_method_maps: 
        valid_keys_for_loading.update(m_map.keys())
    
    actual_keys_to_process_loading = [key for key in method_keys_to_load if key in valid_keys_for_loading]
    if len(actual_keys_to_process_loading) != len(method_keys_to_load):
        skipped_keys = set(method_keys_to_load) - set(actual_keys_to_process_loading)
        print(f"INFO: The following method keys specified for loading were not found in any provided METHOD_PROMPT_MAP and will be skipped for data loading: {sorted(list(skipped_keys))}")

    for dataset_name_table in allowed_datasets:
        for method_key in actual_keys_to_process_loading: 
            stats["files_checked"] += 1
            score, n_s = None, 0 
            
            prefix = "AI_llama" if "llama" in model_to_load.lower() else "AI_qwen"
            fname_csv = f"{prefix}-{dataset_name_table}-{model_to_load}-{method_key}-n{n_val_param}-wait{wait_val_param}-scores.csv"
            fpath_csv = os.path.join(scores_dir_path, fname_csv)

            if os.path.exists(fpath_csv):
                try:
                    df_score = pd.read_csv(fpath_csv, header=None, index_col=0)
                    target_csv_index = f"{method_key}-n{n_val_param}" 

                    if not df_score.empty and target_csv_index in df_score.index:
                        if 1 in df_score.columns: 
                            raw_score = df_score.loc[target_csv_index, 1]
                            score_val = round(float(raw_score) * 100, 1) 
                            n_s = 0 
                            
                            print(f"  CSV READ: {model_to_load}/{dataset_name_table}/{method_key}. Score: {score_val:.1f}")
                            stats["csv_read"] += 1
                            score = score_val
                        else:
                            print(f"    WARN: Score column (expected as '1') not found in CSV file {fpath_csv} for index '{target_csv_index}'.")
                            stats["errors_reading_csv"] += 1
                    else:
                        if df_score.empty:
                             print(f"    WARN: CSV file {fpath_csv} is empty.")
                        else: 
                             print(f"    WARN: Index '{target_csv_index}' not found in CSV file {fpath_csv}. Available indices: {df_score.index.tolist()}")
                        stats["errors_reading_csv"] += 1
                except KeyError as ke: 
                    print(f"    WARN: Index '{target_csv_index}' caused KeyError in CSV file {fpath_csv}: {ke}")
                    stats["errors_reading_csv"] += 1
                except Exception as e:
                    print(f"    ERROR reading or processing CSV file {fpath_csv}: {e}")
                    stats["errors_reading_csv"] += 1
            else:
                print(f"    INFO: CSV score file NOT FOUND: {fpath_csv}")
                stats["missing_csvs"] += 1
            
            table_type_for_cat = 2 if method_key in METHOD_PROMPT_MAP_TABLE2 else 1
            category_for_df = get_method_category(method_key, table_type=table_type_for_cat)

            all_data.append({'llm': model_to_load, 
                             'dataset': dataset_name_table, 
                             'method': method_key,
                             'category': category_for_df, 
                             'score': score, 
                             'n_samples': n_s})
    
    print(f"\nFinished loading scores. Files Checked: {stats['files_checked']}, CSVs Read: {stats['csv_read']}, Missing CSVs: {stats['missing_csvs']}, CSV Errors: {stats['errors_reading_csv']}")
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df['model_family'] = df['llm'].map(MODEL_FAMILY_MAP); df['model_size'] = df['llm'].map(MODEL_SIZE_MAP)
    return df

# --- Main LaTeX Table Generation Logic (Adapted) ---
def generate_prompt_focused_latex_table(scores_df, target_model_name, current_method_prompt_map,
                                        current_ordered_method_keys, allowed_datasets, dataset_display_map,
                                        table_caption, table_label, table_headers,
                                        wrap_general_prompt_with_inlineinput=True, 
                                        use_minipage_for_structured_prompts=False): 
    if scores_df.empty: return f"% No data to generate table ({table_label}). Scores DataFrame is empty.\n"
    model_scores_df = scores_df[scores_df['llm'] == target_model_name].copy()
    if model_scores_df.empty: return f"% No data for model {target_model_name} in scores DataFrame for table {table_label}.\n"

    final_latex_string = io.StringIO()
    final_latex_string.write(f"% --- Table: {table_label} ---\n") 
    final_latex_string.write(r"\begin{table*}[htbp!]" + "\n") 
    final_latex_string.write(r"\centering" + "\n")
    final_latex_string.write(f"\\caption{{{table_caption}}}\n")
    final_latex_string.write(f"\\label{{{table_label}}}\n")
    final_latex_string.write(TABLE_FONT_SIZE + "\n") 

    current_table_type = 2 if use_minipage_for_structured_prompts else 1

    if current_table_type == 2: # Table 2 (Structured with minipage)
        category_col_width_val = "0.07\\textwidth" 
        prompt_col_width_val = "0.53\\textwidth"   
        prompt_col_char = 'm' 
    else: # Table 1 (General)
        category_col_width_val = "0.09\\textwidth" 
        prompt_col_width_val = "0.515\\textwidth"  
        prompt_col_char = 'p' 
    
    col_spec = f"m{{{category_col_width_val}}}{prompt_col_char}{{{prompt_col_width_val}}}" + ('c' * len(allowed_datasets)) 
    
    final_latex_string.write(r"\begin{tabular}{" + col_spec + "}\n")
    final_latex_string.write(r"\toprule" + "\n")
    latex_header_cells = [rf"\textbf{{{header}}}" for header in table_headers] + \
                         [rf"\textbf{{{escape_latex(dataset_display_map.get(ds_key, ds_key.upper()))}}}" for ds_key in allowed_datasets]
    final_latex_string.write(" & ".join(latex_header_cells) + r" \\" + "\n")
    final_latex_string.write(r"\midrule" + "\n")
    
    max_scores_per_dataset = {}
    if BOLD_MAX_SCORE_PER_DATASET_COLUMN:
        for ds_key in allowed_datasets:
            relevant_scores = model_scores_df[model_scores_df['method'].isin(current_ordered_method_keys) & (model_scores_df['dataset'] == ds_key)]['score']
            dataset_scores = pd.to_numeric(relevant_scores, errors='coerce')
            if not dataset_scores.empty and dataset_scores.notna().any(): max_scores_per_dataset[ds_key] = dataset_scores.max()
            else: max_scores_per_dataset[ds_key] = -1 

    # Pre-calculate category details for \multirow and \hline placement
    row_category_details = []
    if current_ordered_method_keys:
        method_categories_plain = [get_method_category(mk, table_type=current_table_type) for mk in current_ordered_method_keys]
        
        idx = 0
        num_methods = len(current_ordered_method_keys)
        while idx < num_methods:
            current_cat_plain = method_categories_plain[idx]
            count = 0
            temp_idx = idx
            while temp_idx < num_methods and method_categories_plain[temp_idx] == current_cat_plain:
                count += 1
                temp_idx += 1
            
            for k_in_group in range(count):
                row_details = {
                    'method_key': current_ordered_method_keys[idx + k_in_group],
                    'category_name_plain': current_cat_plain, # Plain text for logic
                    'is_first_in_group': (k_in_group == 0),
                    'group_span': count if (k_in_group == 0) else 0,
                    'is_last_in_group': (k_in_group == count - 1)
                }
                row_category_details.append(row_details)
            idx += count


    num_rows_to_print = len(row_category_details)
    for i, details in enumerate(row_category_details):
        method_key = details['method_key']
        if method_key not in current_method_prompt_map:
            print(f"WARNING: Key '{method_key}' in ordered list for table '{table_label}' not in its map. Skipping.", file=sys.stderr)
            continue 
        
        prompt_data = current_method_prompt_map.get(method_key)
        
        row_cells = [] 
        
        if details['is_first_in_group']:
            # For Table 2, get_method_category now returns plain text.
            # For multi-line in \multirow, the LaTeX itself needs \\ if that's desired.
            # User example for Table 2 has single line category names like "None", "User".
            # So, always escape the plain text category name.
            multirow_category_text = escape_latex(details['category_name_plain'])
            row_cells.append(rf"\multirow{{{details['group_span']}}}{{*}}{{{multirow_category_text}}}")
        else:
            row_cells.append("") 

        final_prompt_cell_content = ""
        if use_minipage_for_structured_prompts: # Logic for Table 2
            if not isinstance(prompt_data, dict):
                print(f"ERROR ({table_label}): Prompt data for key '{method_key}' is not a dictionary. Skipping prompt.", file=sys.stderr)
                final_prompt_cell_content = f"Error: Bad prompt data for {method_key}"
            else:
                user_part = prompt_data.get("user_content", "")
                assistant_prefix_part = prompt_data.get("assistant_prefix", "")
                figure_input_text = prompt_data.get("assistant_figure_input")
                assistant_suffix_part = prompt_data.get("assistant_suffix", "") 

                assistant_combined_part = assistant_prefix_part.strip() 
                if figure_input_text:
                    assistant_combined_part += rf" \figureinput{{{figure_input_text.strip()}}}" 
                if assistant_suffix_part.strip(): 
                    assistant_combined_part += " " + assistant_suffix_part.strip()
                
                assembled_prompt = rf"\inlineinputt{{{user_part.strip()}}} \\ \inlineoutputt{{{assistant_combined_part.strip()}}}"
                final_prompt_cell_content = rf"\begin{{minipage}}[c]{{\linewidth}}{{{assembled_prompt}}}\end{{minipage}}"
        else: # Logic for Table 1
            base_display_text_t1 = prompt_data if prompt_data else method_key 
            escaped_text_t1 = escape_latex(base_display_text_t1)
            if wrap_general_prompt_with_inlineinput: 
                final_prompt_cell_content = rf"\inlineinput{{{escaped_text_t1}}}"
            else:
                final_prompt_cell_content = escaped_text_t1
        row_cells.append(final_prompt_cell_content)

        for ds_key in allowed_datasets:
            score_entry = model_scores_df[(model_scores_df['method'] == method_key) & (model_scores_df['dataset'] == ds_key)]
            score_val = score_entry['score'].iloc[0] if not score_entry.empty and pd.notna(score_entry['score'].iloc[0]) else np.nan
            formatted_score = format_score_for_table(score_val)
            if BOLD_MAX_SCORE_PER_DATASET_COLUMN and pd.notna(score_val) and ds_key in max_scores_per_dataset and np.isclose(score_val, max_scores_per_dataset[ds_key]) and formatted_score != '-':
                formatted_score = rf"\textbf{{{formatted_score}}}"
            row_cells.append(formatted_score)
            
        final_latex_string.write(" & ".join(row_cells) + r" \\" + "\n")
        
        # Add \hline if this row is the last of its category group AND it's not the last row of the table
        if details['is_last_in_group'] and i < num_rows_to_print - 1:
            final_latex_string.write(r"\hline" + "\n")
        
    final_latex_string.write(r"\bottomrule" + "\n")
    final_latex_string.write(r"\end{tabular}" + "\n")
    final_latex_string.write(r"\end{table*}" + "\n\n% --- End of Table ---\n")
    return final_latex_string.getvalue()

# --- Key Validation Function ---
def validate_configuration(ordered_keys, method_map, table_name):
    print(f"\n--- Validating Configuration for {table_name} ---")
    is_valid = True
    map_keys_set = set(method_map.keys())
    
    processed_ordered_keys = []
    seen_in_ordered = set()
    for key in ordered_keys:
        if key in seen_in_ordered:
            print(f"ERROR ({table_name}): Duplicate key '{key}' found in ORDERED_METHOD_KEYS. Please ensure all keys in this list are unique.", file=sys.stderr)
            is_valid = False
        else:
            seen_in_ordered.add(key)
            processed_ordered_keys.append(key)

    for key in processed_ordered_keys: 
        if key not in map_keys_set:
            print(f"ERROR ({table_name}): Key '{key}' in ORDERED_METHOD_KEYS is not defined in its METHOD_PROMPT_MAP. Please correct.", file=sys.stderr)
            is_valid = False 

    unused_map_keys = map_keys_set - seen_in_ordered
    if unused_map_keys:
        print(f"INFO ({table_name}): The following keys are in its METHOD_PROMPT_MAP but not in its ORDERED_METHOD_KEYS (will not be in table): {sorted(list(unused_map_keys))}", file=sys.stdout)
    
    if not ordered_keys: 
        print(f"WARNING ({table_name}): ORDERED_METHOD_KEYS list is empty. No rows will be generated for this table.", file=sys.stderr)

    return is_valid

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Starting Script for Model: {TARGET_MODEL_NAME} ---")
    if not os.path.isdir(SCORES_DIR): 
        print(f"ERROR: SCORES_DIR '{SCORES_DIR}' not found.", file=sys.stderr); sys.exit(1)

    valid_config_t1 = validate_configuration(ORDERED_METHOD_KEYS_TABLE1, METHOD_PROMPT_MAP_TABLE1, "Table 1 (General Prompts)")
    valid_config_t2 = validate_configuration(ORDERED_METHOD_KEYS_TABLE2, METHOD_PROMPT_MAP_TABLE2, "Table 2 (LaTeX Prompts)")

    if not (valid_config_t1 and valid_config_t2):
        print("\nExiting due to configuration validation errors. Please check messages above.", file=sys.stderr)
        sys.exit(1)

    all_required_keys = list(set(ORDERED_METHOD_KEYS_TABLE1) | set(ORDERED_METHOD_KEYS_TABLE2))
    if not all_required_keys:
        print("No methods specified in any table configuration. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    print(f"\nINFO: Will attempt to load data for the following unique method keys: {sorted(all_required_keys)}")

    scores_df = load_data_and_calculate_scores(
        SCORES_DIR, TARGET_MODEL_NAME, ALLOWED_DATASETS, 
        all_required_keys, N_VAL_TABLE, WAIT_VAL_TABLE,
        all_method_maps=[METHOD_PROMPT_MAP_TABLE1, METHOD_PROMPT_MAP_TABLE2]
    )

    if scores_df.empty and all_required_keys: 
        print("Exiting: No data loaded or calculated overall. Check CSV score files and paths.", file=sys.stderr)
        sys.exit(1)
    
    latex_preamble_string = io.StringIO()
    latex_preamble_string.write("% --- Suggested LaTeX Preamble (ensure these are in your main .tex file) ---\n")
    latex_preamble_string.write("% \\documentclass{article}\n")
    latex_preamble_string.write("% \\usepackage[margin=0.5in]{geometry} % Adjust as needed\n")
    latex_preamble_string.write("% \\usepackage{booktabs} % For \toprule, \midrule, \bottomrule\n")
    latex_preamble_string.write("% \\usepackage{multirow} % For multirow cells (used for Category)\n")
    latex_preamble_string.write("% \\usepackage{amsmath}\n")
    latex_preamble_string.write("% \\usepackage{caption}\n")
    latex_preamble_string.write("% \\usepackage{array}   % For m{}, p{} column types\n")
    latex_preamble_string.write("% \\usepackage{longtable}\n")
    latex_preamble_string.write("% % For Table 1 (if using inlineinput for simple prompts):\n")
    latex_preamble_string.write("% \\newcommand{\\inlineinput}[1]{#1} % Basic definition. Customize if needed.\n")
    latex_preamble_string.write("% % For Table 2 (if using structured prompts with custom commands like usergrayprompt, figureinput, etc.):\n")
    latex_preamble_string.write("% % You'll need definitions for: \\inlineinputt, \\inlineoutputt, \\figureinput, \\usergrayprompt, \\assistantblueprompt, etc.\n")
    latex_preamble_string.write("% % Ensure these commands (especially those with text content like \\figureinput) handle text wrapping correctly.\n")
    latex_preamble_string.write("% \\begin{document}\n\n")
    full_latex_output = latex_preamble_string.getvalue()


    # --- Generate and Save First Table ---
    if ORDERED_METHOD_KEYS_TABLE1:
        print("\n\n--- Generating First Table (General Prompts) ---")
        caption_t1 = f"Effect of phrase types on Macro F1 scores (\%) for {escape_latex(TARGET_MODEL_NAME)}." 
        label_t1 = f"tab:prompt_performance_{TARGET_MODEL_NAME.replace('-', '_')}_general_nested_cat" 
        headers_t1 = ["Category", "Prompt Phrase"]
        latex_output_t1 = generate_prompt_focused_latex_table(
            scores_df, TARGET_MODEL_NAME, METHOD_PROMPT_MAP_TABLE1, ORDERED_METHOD_KEYS_TABLE1,
            ALLOWED_DATASETS, DATASET_DISPLAY_MAP_TABLE, caption_t1, label_t1, headers_t1,
            wrap_general_prompt_with_inlineinput=True, 
            use_minipage_for_structured_prompts=False 
        )
        full_latex_output += latex_output_t1
        print("\n" + "="*25 + " Generated LaTeX Code (First Table) " + "="*25)
        print(latex_output_t1) 
        print("="*70)
    else:
        print("\nINFO: Skipping generation of Table 1 as its ORDERED_METHOD_KEYS_TABLE1 is empty.")

    # --- Generate and Save Second Table ---
    if ORDERED_METHOD_KEYS_TABLE2:
        print("\n\n--- Generating Second Table (Structured LaTeX Prompts) ---")
        caption_t2 = f"Effect of phrase placement on Macro F1 scores (\%) for {escape_latex(TARGET_MODEL_NAME)}." 
        label_t2 = f"tab:prompt_performance_{TARGET_MODEL_NAME.replace('-', '_')}_structured_minipage_tt_nested_cat" 
        headers_t2 = ["Placement", "Full Prompt"]
        latex_output_t2 = generate_prompt_focused_latex_table(
            scores_df, TARGET_MODEL_NAME, METHOD_PROMPT_MAP_TABLE2, ORDERED_METHOD_KEYS_TABLE2,
            ALLOWED_DATASETS, DATASET_DISPLAY_MAP_TABLE, caption_t2, label_t2, headers_t2,
            wrap_general_prompt_with_inlineinput=False, 
            use_minipage_for_structured_prompts=True      
        )
        full_latex_output += "\n\n" + latex_output_t2
        print("\n" + "="*25 + " Generated LaTeX Code (Second Table) " + "="*25)
        print(latex_output_t2) 
        print("="*70)
    else:
        print("\nINFO: Skipping generation of Table 2 as its ORDERED_METHOD_KEYS_TABLE2 is empty.")

    full_latex_output += "\n% \\end{document}\n"

    # Ensure the TABLES_DIR from config exists
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True) # <--- ADD DIRECTORY CREATION

    combined_output_filename_only = f"{TARGET_MODEL_NAME.replace('.', '_')}_combined_prompt_tables.tex" # Or a more descriptive name
    combined_output_filepath = config.TABLES_DIR / combined_output_filename_only # <--- CONSTRUCT FULL PATH

    try:
        with open(combined_output_filepath, 'w', encoding='utf-8') as f: # <--- USE combined_output_filepath
            f.write(full_latex_output)
        print(f"\nCombined LaTeX code for both tables saved to: {combined_output_filepath}") # <--- USE combined_output_filepath
    except IOError as e:
        print(f"\nError saving combined LaTeX file: {e}", file=sys.stderr)


    print("\n--- Script Finished ---")
    print("\nOverall Notes:")
    print(r"- Scores are Macro F1 (%).")
    print(r"- Tables now feature a nested 'Category'/'Placement' column as the first column, vertically centered using an 'm' column type.")
    print(r"- An \hline is added after each category group in both tables (except after the very last group).")
    print(r"- Table 1 prompts are wrapped with \inlineinput{...} (if defined in your preamble).")
    print(r"- Table 2 prompts are structured using your \inlineinputt{} and \inlineoutputt{} commands, all within a \begin{minipage}[c]{\linewidth}...\end{minipage}.")
    print(r"  Category names for Table 2 are now single line as per your example ('None', 'User', 'Assistant').")
    print(r"  Ensure \inlineinputt, \inlineoutputt, \figureinput, and any other custom LaTeX commands/environments are correctly defined in your preamble and handle text wrapping appropriately.")
    if BOLD_MAX_SCORE_PER_DATASET_COLUMN: print("- Bolding is applied to the max F1 score in each dataset column within each table.")
    print(f"- F1 score results are NOT cached. Scores are read directly from CSVs in '{SCORES_DIR}' on each run.")
    print(f"- Ensure your CSV score files are in '{SCORES_DIR}' and follow the expected naming convention.")