import os
import pandas as pd
import io
import re
import numpy as np
import sys
from collections import defaultdict
# sklearn.metrics.f1_score is no longer needed as scores are read from CSV

# --- Seed for reproducibility ---
np.random.seed(0)

# --- Configuration ---
# SCORES_DIR: Directory where the .csv score files are located.
SCORES_DIR = '/data3/zkachwal/visual-reasoning/data/ai-generation/scores/' # Please update this path if necessary
# CACHE_DIR: This directory will not be actively used as caching is disabled for CSV loading.
CACHE_DIR = "./csv_score_cache_unused/" 

# Define all models to be included (for parsing and potential other uses)
# "CoDE" removed from this list
ALL_MODELS_CONFIG = [
    "qwen25-3b", "qwen25-7b", "qwen25-32b", "qwen25-72b",
    # "CoDE" 
]
CODE_MODEL_NAME = "CoDE" # Constant for CoDE identifier (defined but not used for table generation if not in ALL_MODELS_CONFIG)

# Define datasets (will become columns in the table)
ALLOWED_DATASETS = ["d32k", "df402k","genimage2k"]
# Define methods for LLMs (will be rows under each LLM model size)
LLM_METHODS_ORDER = ["zeroshot", "zeroshot-cot", "zeroshot-2-artifacts"]
BASELINE_METHOD_FOR_DIFF = "zeroshot-cot" 
TARGET_METHOD_FOR_DIFF = "zeroshot-2-artifacts"


# Define other parameters used in filename construction (for CSVs and potentially other uses)
N_VAL_TABLE = "1"
WAIT_VAL_TABLE = "0"

# --- Model Parsing and Family Configuration ---
MODEL_FAMILIES_DETECTED = []
MODEL_FAMILY_MAP = {}
MODEL_SIZE_MAP = {}
MODELS_BY_FAMILY = defaultdict(list)
CODE_FAMILY_NAME = "CoDE" # Defined but CoDE family won't be in TABLE_MODEL_FAMILIES_ORDER
CODE_SIZE_IDENTIFIER = "6m" 

print("--- Parsing Model Names and Families ---")
for model_name_config in ALL_MODELS_CONFIG: # This loop will not process "CoDE" if it's removed above
    if model_name_config == CODE_MODEL_NAME: # This block will likely not be hit
        family_name = CODE_FAMILY_NAME
        size = CODE_SIZE_IDENTIFIER 
        if family_name not in MODEL_FAMILIES_DETECTED:
            MODEL_FAMILIES_DETECTED.append(family_name)
        MODEL_FAMILY_MAP[model_name_config] = family_name
        MODEL_SIZE_MAP[model_name_config] = size
        MODELS_BY_FAMILY[family_name].append(model_name_config)
    else: # LLM parsing
        match = re.match(r'([a-zA-Z]+)(\d+(\.\d+)?)?-(\d+[bB])', model_name_config)
        if match:
            base_name = match.group(1)
            version = match.group(2)
            size_str_from_config = match.group(4) 
            size = size_str_from_config.lower() 
            
            family_name = base_name
            if base_name == 'llama': family_name = 'llama3.2' 
            elif version:
                if base_name == 'qwen' and version == '25': family_name = 'qwen2.5'
                else: family_name = f"{base_name}{version}"
            if family_name not in MODEL_FAMILIES_DETECTED:
                MODEL_FAMILIES_DETECTED.append(family_name)
            MODEL_FAMILY_MAP[model_name_config] = family_name
            MODEL_SIZE_MAP[model_name_config] = size 
            MODELS_BY_FAMILY[family_name].append(model_name_config)
        else:
            print(f"Warning: Could not parse family/size from LLM name: {model_name_config}", file=sys.stderr)
            MODEL_FAMILY_MAP[model_name_config] = "unknown_llm_family"
            MODEL_SIZE_MAP[model_name_config] = "unknown_llm_size"
            MODELS_BY_FAMILY["unknown_llm_family"].append(model_name_config)
            if "unknown_llm_family" not in MODEL_FAMILIES_DETECTED:
                MODEL_FAMILIES_DETECTED.append("unknown_llm_family")
# Also parse CoDE if it was in ALL_MODELS_CONFIG initially, even if not used for table
if CODE_MODEL_NAME not in MODEL_FAMILY_MAP and CODE_MODEL_NAME in ALL_MODELS_CONFIG: 
    MODEL_FAMILY_MAP[CODE_MODEL_NAME] = CODE_FAMILY_NAME
    MODEL_SIZE_MAP[CODE_MODEL_NAME] = CODE_SIZE_IDENTIFIER
    MODELS_BY_FAMILY[CODE_FAMILY_NAME].append(CODE_MODEL_NAME)
    if CODE_FAMILY_NAME not in MODEL_FAMILIES_DETECTED:
         MODEL_FAMILIES_DETECTED.append(CODE_FAMILY_NAME)

print("--- End Model Parsing ---")

# CoDE family removed from this list. Llama3.2 is kept for parsing but table will focus on Qwen2.5.
TABLE_MODEL_FAMILIES_ORDER = ['qwen2.5', 'llama3.2'] 
CODE_OVERALL_METHOD_KEY = "Overall" 
TABLE_METHODS_ORDER_LLM = LLM_METHODS_ORDER

METHOD_DISPLAY_NAME_MAPPING = {
    'zeroshot': 'zeroshot',
    'zeroshot-cot': 'zeroshot-cot',
    'zeroshot-2-artifacts': r'zeroshot-s$^2$',
    CODE_OVERALL_METHOD_KEY: "trained on D3" 
}

# --- LaTeX Table Formatting Options ---
BOLD_MAX_SCORE_PER_MODEL_DATASET = True 
ZERO_PAD_SCORE = True 
TABLE_FONT_SIZE = r"\small" 
DATASET_DISPLAY_MAP_TABLE = {
    "d32k": "D3 (2k)",
    "df402k": "DF40 (2k)",
    "genimage2k": "GenImage (2k)"
}

# --- Helper Functions ---
def extract_size_numeric(size_str):
    if size_str == CODE_SIZE_IDENTIFIER: # "6m"
        return 6 * 1e6 
    if size_str == "unknown_llm_size": return float('inf') 
    
    match_b = re.match(r'(\d+(\.\d+)?)b', size_str)
    if match_b:
        return float(match_b.group(1)) * 1e9 

    match_m = re.match(r'(\d+(\.\d+)?)m', size_str) 
    if match_m:
        return float(match_m.group(1)) * 1e6 
    
    match_num = re.match(r'(\d+(\.\d+)?)', size_str) 
    return float(match_num.group(1)) if match_num else float('inf')


def escape_latex(text):
    if text is None: return ''
    text_str = str(text)
    if not text_str: return ''
    text_str = text_str.replace('\\', r'\textbackslash{}')
    replacements = { '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    for char, escaped_char in replacements.items():
        text_str = text_str.replace(char, escaped_char)
    return text_str

def _format_number_for_score(num_val):
    if pd.isna(num_val) or num_val is None:
        return "-"
    formatted_val = f"{float(num_val):.1f}" 
    if ZERO_PAD_SCORE and 0 <= float(num_val) < 10:
        formatted_val = f"0{formatted_val}"
    return formatted_val

def format_score_for_table(score):
    if pd.isna(score) or score is None:
        return '-'
    return _format_number_for_score(score)

# --- Data Loading from CSVs ---
def load_scores_from_csvs(scores_dir_path, all_models_to_load, allowed_datasets_table,
                          llm_methods_table, code_method_key, n_val_param, wait_val_param):
    print(f"\n--- Loading Scores from CSV files in: {scores_dir_path} (Caching Disabled) ---")
    all_data = []
    stats = {"csv_read": 0, "missing_csvs": 0, "errors_reading_csv": 0, "files_checked": 0}

    for model_name in all_models_to_load: 
        current_methods_to_process = [code_method_key] if model_name == CODE_MODEL_NAME else llm_methods_table
        
        for dataset_name_table in allowed_datasets_table:
            for method_key in current_methods_to_process:
                stats["files_checked"] += 1
                score, n_s = None, 0  

                prefix = "AI_util" if "llama" in model_name.lower() else "AI_dev"
                
                if model_name == CODE_MODEL_NAME: 
                    prefix = "AI_dev" 
                    fname_csv = f"{prefix}-{dataset_name_table}-{model_name}-scores.csv"
                else:
                    fname_csv = f"{prefix}-{dataset_name_table}-{model_name}-{method_key}-n{n_val_param}-wait{wait_val_param}-scores.csv"

                fpath_csv = os.path.join(scores_dir_path, fname_csv)

                if os.path.exists(fpath_csv):
                    try:
                        df_score_csv = pd.read_csv(fpath_csv, header=None, index_col=0)
                        
                        if model_name == CODE_MODEL_NAME: 
                            if len(df_score_csv) >= 2 and len(df_score_csv.columns) >= 1:
                                raw_score = df_score_csv.iloc[1, 0] 
                                score_val = round(float(raw_score) * 100, 1)
                                print(f"  CSV READ (CoDE): {model_name}/{dataset_name_table}. Score: {score_val:.1f} from iloc[1,0]")
                                stats["csv_read"] += 1
                                score = score_val
                            else:
                                print(f"    WARN (CoDE): CSV {fpath_csv} does not have expected structure. Shape: {df_score_csv.shape}")
                                stats["errors_reading_csv"] += 1
                        else: 
                            target_csv_index = f"{method_key}-n{n_val_param}" 
                            if not df_score_csv.empty and target_csv_index in df_score_csv.index:
                                if 1 in df_score_csv.columns:  
                                    raw_score = df_score_csv.loc[target_csv_index, 1]
                                    score_val = round(float(raw_score) * 100, 1)
                                    print(f"  CSV READ: {model_name}/{dataset_name_table}/{method_key}. Score: {score_val:.1f}")
                                    stats["csv_read"] += 1
                                    score = score_val
                                else:
                                    print(f"    WARN: Score column ('1') not found in CSV {fpath_csv} for index '{target_csv_index}'.")
                                    stats["errors_reading_csv"] += 1
                            else:
                                if df_score_csv.empty:
                                    print(f"    WARN: CSV file {fpath_csv} is empty.")
                                else:
                                    print(f"    WARN: Index '{target_csv_index}' not found in CSV {fpath_csv}. Available: {df_score_csv.index.tolist()}")
                                stats["errors_reading_csv"] += 1
                    except KeyError as ke: 
                        print(f"    WARN: KeyError during CSV processing {fpath_csv}: {ke}")
                        stats["errors_reading_csv"] += 1
                    except Exception as e:
                        print(f"    ERROR reading/processing CSV {fpath_csv}: {e}")
                        stats["errors_reading_csv"] += 1
                else:
                    print(f"    INFO: CSV score file NOT FOUND: {fpath_csv}")
                    stats["missing_csvs"] += 1
                
                all_data.append({
                    'llm': model_name, 'dataset': dataset_name_table, 'method': method_key,
                    'score': score, 'n_samples': n_s 
                })
    
    print(f"\nFinished loading scores. Files Checked: {stats['files_checked']}, CSVs Read: {stats['csv_read']}, Missing CSVs: {stats['missing_csvs']}, CSV Errors: {stats['errors_reading_csv']}")
    if not all_data: return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df['model_family'] = df['llm'].map(MODEL_FAMILY_MAP)
    df['model_size'] = df['llm'].map(MODEL_SIZE_MAP) 
    df['parameter_size_numeric'] = df['model_size'].apply(extract_size_numeric)
    
    # Filter out families not in TABLE_MODEL_FAMILIES_ORDER before setting categorical type
    df = df[df['model_family'].isin(TABLE_MODEL_FAMILIES_ORDER)]

    family_cat_type = pd.CategoricalDtype(categories=TABLE_MODEL_FAMILIES_ORDER, ordered=True)
    df['model_family_ordered'] = df['model_family'].astype(family_cat_type)
    
    df = df.sort_values(by=['model_family_ordered', 'parameter_size_numeric', 'llm', 'method', 'dataset'])
    df = df.drop(columns=['model_family_ordered']) 
    return df

# --- Main LaTeX Table Generation Logic ---
def generate_latex_table(scores_df_table):
    if scores_df_table.empty: return "% No data to generate table.\n"
    final_latex_string = io.StringIO()
    final_latex_string.write("% --- Suggested LaTeX Preamble ---\n")
    final_latex_string.write("% \\documentclass{article}\n")
    final_latex_string.write("% \\usepackage[margin=1in]{geometry}\n")
    final_latex_string.write("% \\usepackage{booktabs}\n")
    final_latex_string.write("% \\usepackage{multirow}\n")
    final_latex_string.write("% \\usepackage{amsmath}\n")
    final_latex_string.write("% \\usepackage{caption}\n")
    final_latex_string.write("% \\usepackage{amsfonts}\n")
    final_latex_string.write("% \\begin{document}\n\n")
    
    final_latex_string.write("% --- Main Performance Table (Qwen2.5 focus) ---\n")
    final_latex_string.write(r"\begin{table*}[htbp!]" + "\n")
    final_latex_string.write(r"\centering" + "\n")
    # Updated Caption
    final_latex_string.write(f"\\caption{{Effect of prompts on Macro F1 scores (\%) across Qwen2.5 model sizes}}\n")
    final_latex_string.write(f"\\label{{tab:qwen25_prompt_effects}}\n") # Updated Label
    final_latex_string.write(TABLE_FONT_SIZE + "\n")

    num_dataset_cols = len(ALLOWED_DATASETS)
    total_cols = 3 + num_dataset_cols 
    col_spec = 'lll' + ('l' * num_dataset_cols) 
    final_latex_string.write(r"\begin{tabular}{" + col_spec + "}\n")
    final_latex_string.write(r"\toprule" + "\n")

    header_cells = [r"\textbf{Model Family}", r"\textbf{Model Size}", r"\textbf{Method}"]
    for ds_key in ALLOWED_DATASETS:
        display_name = DATASET_DISPLAY_MAP_TABLE.get(ds_key, ds_key.upper())
        header_cells.append(rf"\textbf{{{escape_latex(display_name)}}}")
    final_latex_string.write(" & ".join(header_cells) + r" \\" + "\n")
    final_latex_string.write(r"\midrule" + "\n")
    
    # --- Filter for Qwen2.5 family for this specific table output ---
    qwen_family_name_to_display = 'qwen2.5'
    family_data_all_models = scores_df_table[scores_df_table['model_family'] == qwen_family_name_to_display].copy()

    if family_data_all_models.empty:
        print(f"Warning: No data found for the '{qwen_family_name_to_display}' model family. Table will be empty.", file=sys.stderr)
        final_latex_string.write(f"% No data available for {qwen_family_name_to_display} to generate table.\n")
        final_latex_string.write(r"\bottomrule" + "\n")
        final_latex_string.write(r"\end{tabular}" + "\n")
        final_latex_string.write(r"\end{table*}" + "\n\n% --- End of Table ---\n")
        return final_latex_string.getvalue()

    # is_code_family will always be False here as we are only processing qwen2.5
    is_code_family = False 
    
    sorted_llms_in_family = sorted(
        MODELS_BY_FAMILY.get(qwen_family_name_to_display, []), 
        key=lambda llm_name: extract_size_numeric(MODEL_SIZE_MAP.get(llm_name, "unknown_llm_size"))
    )
    
    if not sorted_llms_in_family:
        print(f"Warning: No models configured for the '{qwen_family_name_to_display}' family. Table will be empty.", file=sys.stderr)
        # ... (error handling as before)
        final_latex_string.write(f"% No models configured for {qwen_family_name_to_display}.\n")
        final_latex_string.write(r"\bottomrule" + "\n")
        final_latex_string.write(r"\end{tabular}" + "\n")
        final_latex_string.write(r"\end{table*}" + "\n\n% --- End of Table ---\n")
        return final_latex_string.getvalue()


    methods_for_family_display = TABLE_METHODS_ORDER_LLM # Qwen uses LLM methods
    
    num_rows_fam_display = 0
    for llm_name_calc in sorted_llms_in_family:
        if not family_data_all_models[family_data_all_models['llm'] == llm_name_calc].empty:
             num_rows_fam_display += len(methods_for_family_display)
    
    if num_rows_fam_display == 0:
        print(f"Warning: No data rows to display for '{qwen_family_name_to_display}' models. Table will be effectively empty.", file=sys.stderr)
        # ... (error handling as before)
        final_latex_string.write(f"% No data rows for {qwen_family_name_to_display} models.\n")
        final_latex_string.write(r"\bottomrule" + "\n")
        final_latex_string.write(r"\end{tabular}" + "\n")
        final_latex_string.write(r"\end{table*}" + "\n\n% --- End of Table ---\n")
        return final_latex_string.getvalue()


    family_disp_name = escape_latex(qwen_family_name_to_display)
    first_llm_in_fam = True 

    for llm_idx, current_llm_name_for_data in enumerate(sorted_llms_in_family):
        model_specific_data = family_data_all_models[family_data_all_models['llm'] == current_llm_name_for_data]
        if model_specific_data.empty: continue

        current_model_size_str = MODEL_SIZE_MAP.get(current_llm_name_for_data, "") 

        if not first_llm_in_fam : # Only add cmidrule if there are multiple qwen sizes
             final_latex_string.write(rf"\cmidrule(lr){{2-{total_cols}}}" + "\n")
        
        max_scores_bold = {} 
        # BOLD_MAX_SCORE_PER_MODEL_DATASET applies to LLMs
        for ds_key_bold in ALLOWED_DATASETS:
            current_max = -1.0
            for m_key_bold in methods_for_family_display:
                s_entry = model_specific_data[(model_specific_data['method'] == m_key_bold) & (model_specific_data['dataset'] == ds_key_bold)]['score']
                if not s_entry.empty and pd.notna(s_entry.iloc[0]):
                    current_max = max(current_max, s_entry.iloc[0])
            if current_max > -1.0: max_scores_bold[ds_key_bold] = current_max
        
        for method_idx, method_key_row in enumerate(methods_for_family_display):
            row_cells = []
            if llm_idx == 0 and method_idx == 0: 
                if num_rows_fam_display > 0 : 
                     row_cells.append(rf"\multirow{{{num_rows_fam_display}}}{{*}}{{{family_disp_name}}}")
                else: 
                     row_cells.append(family_disp_name)
            else:
                row_cells.append("")

            if method_idx == 0:
                raw_size_str_for_display = MODEL_SIZE_MAP.get(current_llm_name_for_data, "")
                size_disp_content = raw_size_str_for_display
                if size_disp_content and size_disp_content.endswith('b'): 
                    size_disp_content = size_disp_content[:-1] + 'B'
                # No "6M" display logic needed here as we are focused on Qwen2.5
                    
                size_disp_name = "-" if not size_disp_content else escape_latex(size_disp_content)
                row_cells.append(rf"\multirow{{{len(methods_for_family_display)}}}{{*}}{{{size_disp_name}}}")
            else:
                row_cells.append("")
            
            method_disp_raw = METHOD_DISPLAY_NAME_MAPPING.get(method_key_row, method_key_row)
            method_disp_final = method_disp_raw 
            row_cells.append(method_disp_final)

            for ds_key_cell in ALLOWED_DATASETS:
                data_entry = model_specific_data[(model_specific_data['method'] == method_key_row) & (model_specific_data['dataset'] == ds_key_cell)]
                score_val_cell = data_entry['score'].iloc[0] if not data_entry.empty and pd.notna(data_entry['score'].iloc[0]) else np.nan
                
                score_str_part = format_score_for_table(score_val_cell)
                
                display_score_part = score_str_part
                # Bolding applies as is_code_family is False
                if BOLD_MAX_SCORE_PER_MODEL_DATASET and \
                   pd.notna(score_val_cell) and ds_key_cell in max_scores_bold and \
                   np.isclose(score_val_cell, max_scores_bold[ds_key_cell]) and \
                   score_str_part != '-':
                    display_score_part = rf"\textbf{{{score_str_part}}}"
                
                cell_content = display_score_part
                
                # Diff calculation applies as is_code_family is False
                if method_key_row == TARGET_METHOD_FOR_DIFF and pd.notna(score_val_cell):
                    baseline_entry = model_specific_data[(model_specific_data['method'] == BASELINE_METHOD_FOR_DIFF) & (model_specific_data['dataset'] == ds_key_cell)]
                    baseline_score_val = baseline_entry['score'].iloc[0] if not baseline_entry.empty and pd.notna(baseline_entry['score'].iloc[0]) else np.nan
                    
                    if pd.notna(baseline_score_val):
                        diff = score_val_cell - baseline_score_val
                        formatted_diff = f"{diff:+.1f}" 
                        diff_suffix = f" \\mbox{{\\tiny ({formatted_diff})}}"
                        cell_content += diff_suffix
                        
                row_cells.append(cell_content)
            final_latex_string.write(" & ".join(row_cells) + r" \\" + "\n")
        first_llm_in_fam = False 

    final_latex_string.write(r"\bottomrule" + "\n")
    final_latex_string.write(r"\end{tabular}" + "\n")
    final_latex_string.write(r"\end{table*}" + "\n\n% --- End of Table ---\n")
    return final_latex_string.getvalue()

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.isdir(SCORES_DIR):
        print(f"ERROR: SCORES_DIR '{SCORES_DIR}' not found. Please check the path.", file=sys.stderr)
        sys.exit(1) 

    scores_df_main = load_scores_from_csvs(
        SCORES_DIR,
        ALL_MODELS_CONFIG, 
        ALLOWED_DATASETS,
        LLM_METHODS_ORDER,
        CODE_OVERALL_METHOD_KEY, 
        N_VAL_TABLE,
        WAIT_VAL_TABLE
    )

    if scores_df_main.empty and len(ALL_MODELS_CONFIG) > 0: 
        non_code_models_configured = any(model != CODE_MODEL_NAME for model in ALL_MODELS_CONFIG)
        if non_code_models_configured or (CODE_MODEL_NAME in ALL_MODELS_CONFIG): 
             print("Exiting: No data loaded from CSVs for the configured models. Check CSV files, paths, and naming conventions.", file=sys.stderr)
             sys.exit(1)
        else: 
             print("No models configured in ALL_MODELS_CONFIG. Exiting.")
             sys.exit(1)
    
    latex_output_main = generate_latex_table(scores_df_main)

    print("\n" + "="*25 + " Generated LaTeX Code " + "="*25)
    print(latex_output_main)
    print("="*70)

    output_filename_main = "qwen25_prompt_effects_table.tex" 
    try:
        with open(output_filename_main, 'w', encoding='utf-8') as f:
            f.write(latex_output_main)
        print(f"\nLaTeX table code saved to: {output_filename_main}")
    except IOError as e:
        print(f"\nError saving LaTeX to file {output_filename_main}: {e}", file=sys.stderr)

    print("\nNotes:")
    print(r"- Scores are Macro F1 (%) read directly from CSV files.")
    print(f"- The table displays results specifically for the Qwen2.5 model family.")
    print(f"- For '{METHOD_DISPLAY_NAME_MAPPING.get(TARGET_METHOD_FOR_DIFF, TARGET_METHOD_FOR_DIFF)}', values in parentheses (e.g., \\tiny (+2.5)) show the difference from the '{METHOD_DISPLAY_NAME_MAPPING.get(BASELINE_METHOD_FOR_DIFF, BASELINE_METHOD_FOR_DIFF)}' method.")
    print(f"- Scores are read from CSVs in '{SCORES_DIR}'. Caching is disabled.")
    print(f"- Expected CSV naming for LLMs: {{prefix}}-{{dataset}}-{{model}}-{{method}}-n{N_VAL_TABLE}-wait{WAIT_VAL_TABLE}-scores.csv")
    print("- CoDE model is currently excluded from the table.")
    if BOLD_MAX_SCORE_PER_MODEL_DATASET:
        print("- Bolding is applied to the max F1 score (main value only) for each (Model Size, Dataset) across methods within the Qwen2.5 family.")
    print("- LLM model sizes like '3b' are displayed as '3B'. The script is configured to display CoDE model size as '6M' if it were included in a table.")

