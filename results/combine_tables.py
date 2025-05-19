import os
import pandas as pd
import json
import io
import re
import numpy as np
import sys
from collections import defaultdict

# --- Configuration ---

# Data Paths
DF40_FILE = '/data3/singhdan/DF40/10k_sample_df40.csv'
D3_DIR = '/data3/zkachwal/ELSA_D3/'
GENIMAGE_FILE = '/data3/singhdan/genimage/10k_random_sample.csv'
RESPONSES_DIR = '/data3/zkachwal/visual-reasoning/data/ai-generation/responses/'
# SCORES_DIR is not used in this version as all scores come from RESPONSES_DIR's .jsonl files

# Models, Methods, and Datasets
MODELS_ABBR = ['qwen2.5', 'llama3', 'CoDE'] 
MODEL_NAME_MAP_FULL = {'llama3': 'llama3-11b', 'qwen2.5': 'qwen25-7b'} # For LLM .jsonl filenames

# LLM_METHODS applies to LLMs. CoDE will have its own handling for its .jsonl.
LLM_METHODS = ['zeroshot', 'zeroshot-cot', 'zeroshot-2-artifacts'] 
DATASETS = ['d3', 'df40', 'genimage']

# Subset Naming
SUBSET_NAME_MAPPING = {
    "real": "real", "image_gen0": "dfif", "image_gen1": "sd1.4", "image_gen2": "sd2.1", "image_gen3": "sdxl",
    "collabdiff": "cdif", "midjourney": "midj", "stargan": "sgan1", "starganv2": "sgan2", "styleclip": "sclip",
    "whichfaceisreal": "wfir", "adm": "adm", "biggan": "bgan", "glide_copy": "glide",
    "stable_diffusion_v_1_4": "sd1.4", "stable_diffusion_v_1_5": "sd1.5", "vqdm": "vqdm", "wukong": "wk"
}

# Method Name Mapping (for LLMs)
METHOD_NAME_MAPPING = {
    'zeroshot': 'zeroshot',
    'zeroshot-cot': 'zeroshot-cot',
    'zeroshot-2-artifacts': 'zeroshot-s$^2$'
}
METHOD_TO_ADD_OURS_SUFFIX = 'zeroshot-2-artifacts' # Applies to LLM methods

# LaTeX Formatting Options
BOLD_MAX_RECALL_PER_MODEL = True 
ZERO_PAD_RECALL = True 
TABLE_FONT_SIZE = r"\small"
DIFF_FONT_SIZE = r"\scriptsize" 

# Internal identifier for CoDe's "method" when processing its .jsonl recall data
CODE_RECALL_METHOD_KEY = "recall_data" 
CODE_DISPLAY_METHOD_NAME = "trained on D3" # How CoDe's method is named in the table

# --- Helper Functions ---

def load_genimage_data(file):
    """Loads GenImage data mapping image paths to lowercase subset names."""
    if not os.path.exists(file):
        print(f"Warning: GenImage data file not found: {file}", file=sys.stderr)
        return {}
    try:
        data = pd.read_csv(file)
        subset_data = {}
        required_cols = ['dataset', 'img_path']
        if not all(col in data.columns for col in required_cols):
            print(f"Error: GenImage CSV missing required columns: {required_cols}", file=sys.stderr)
            return {}
        for _, row in data.iterrows():
            subset = str(row['dataset']).lower()
            image = row['img_path']
            subset_data[image] = subset
        return subset_data
    except Exception as e:
        print(f"Error loading GenImage data from {file}: {e}", file=sys.stderr)
        return {}

def load_d3_data(file_dir):
    """Loads D3 data mapping image paths to lowercase subset names from filenames."""
    print(f"\n--- Debugging D3 Loading from: {file_dir} ---")
    if not os.path.isdir(file_dir):
        print(f"Warning: D3 data directory not found: {file_dir}", file=sys.stderr)
        return {}
    subset_data = {}
    extracted_subsets = set()
    try:
        files_in_dir = os.listdir(file_dir)
        print(f"Found {len(files_in_dir)} items in D3 directory.")
        processed_png_count = 0
        for file in files_in_dir:
            if file.endswith(".png"):
                processed_png_count += 1
                image_path = os.path.join(file_dir, file)
                base_name = file.replace('.png', '')
                parts = base_name.split('_')
                raw_subset = 'unknown'

                if len(parts) > 1:
                    if 'real' in parts:
                        raw_subset = 'real'
                    else:
                        joined_parts = "_".join(parts[1:])
                        raw_subset_candidate = re.sub(r'_\d+$', '', joined_parts)
                        if raw_subset_candidate:
                            raw_subset = raw_subset_candidate
                        else:
                            print(f"Warning: D3 file '{file}' - Could not determine subset after joining/stripping: {parts}", file=sys.stderr)
                else:
                    print(f"Warning: Could not determine subset pattern for D3 file (too few parts): {file}", file=sys.stderr)
                
                subset = str(raw_subset).lower()
                subset_data[image_path] = subset
                extracted_subsets.add(subset)
    except Exception as e:
        print(f"Error listing or processing D3 directory {file_dir}: {e}", file=sys.stderr)

    print(f"Processed {processed_png_count} PNG files in D3 directory.")
    print(f"Unique subsets extracted by load_d3_data: {sorted(list(extracted_subsets))}")
    print(f"--- End Debugging D3 Loading ---")
    return subset_data

def load_df40_data(file):
    """Loads DF40 data mapping image paths to lowercase subset names."""
    if not os.path.exists(file):
        print(f"Warning: DF40 data file not found: {file}", file=sys.stderr)
        return {}
    try:
        data = pd.read_csv(file)
        subset_data = {}
        required_cols = ['file_path', 'label', 'dataset']
        if not all(col in data.columns for col in required_cols):
            print(f"Error: DF40 CSV missing required columns: {required_cols}", file=sys.stderr)
            return {}
        for _, row in data.iterrows():
            image = row['file_path']
            raw_subset = 'unknown'
            if row['label'] == 'real':
                raw_subset = 'real'
            elif pd.notna(row['dataset']):
                raw_subset = row['dataset']
            else:
                print(f"Warning: Missing dataset value for non-real image {image} in DF40.", file=sys.stderr)
            subset = str(raw_subset).lower()
            subset_data[image] = subset
        return subset_data
    except Exception as e:
        print(f"Error loading DF40 data from {file}: {e}", file=sys.stderr)
        return {}

def get_recall(true_positives, false_negatives):
    """Calculates recall."""
    denominator = true_positives + false_negatives
    if denominator == 0: return 0.0
    recall = true_positives / denominator
    return recall

def escape_latex(text):
    """Escape characters special to LaTeX, avoiding $, ^ for pre-formatted method names."""
    if text is None: return ''
    text = str(text)
    # Specific replacements, order matters for backslash.
    # $ and ^ are not escaped here to allow s$^2$ type entries from METHOD_NAME_MAPPING.
    text = text.replace('\\', r'\textbackslash{}') 
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('#', r'\#')
    text = text.replace('_', r'\_')
    text = text.replace('{', r'\{')
    text = text.replace('}', r'\}')
    text = text.replace('~', r'\textasciitilde{}')
    return text

def format_score(score):
    """Formats recall score for LaTeX table with optional zero padding."""
    if pd.isna(score) or score is None:
        return '-'
    zero_pad = ZERO_PAD_RECALL
    try:
        score_float = float(score)
    except ValueError:
        return '-' 

    formatted_val_base = f"{score_float:.2f}"
    if zero_pad and 0 <= score_float < 10:
        formatted_val_base = f"0{formatted_val_base}"
    return formatted_val_base

# --- Main Processing Logic ---

def main():
    # --- 1. Load Dataset Mappings ---
    print("Loading dataset mappings...")
    dataset_mappings = {
        'df40': load_df40_data(DF40_FILE),
        'd3': load_d3_data(D3_DIR),
        'genimage': load_genimage_data(GENIMAGE_FILE)
    }
    dataset_mappings = {k: v for k, v in dataset_mappings.items() if v and k in DATASETS}
    if not dataset_mappings:
        print("Error: No dataset mappings loaded successfully. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"\nLoaded mappings for datasets: {list(dataset_mappings.keys())}")

    # --- 2. Initialize Data Structures ---
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'subsets': defaultdict(lambda: np.nan), 'raw_counts': defaultdict(lambda: {'TP': 0, 'FN': 0})})))
    active_subsets_per_dataset = defaultdict(set)

    # --- 3. Process Recall Data from Response Files (for ALL models including CoDE) ---
    print("\nProcessing .jsonl response files for recall calculation...")
    recall_files_processed = 0
    recall_files_missing = 0
    recall_errors_parsing = 0
    recall_format_errors = 0

    for model_abbr in MODELS_ABBR:
        for dataset_name in dataset_mappings.keys():
            current_dataset_mapping = dataset_mappings[dataset_name]
            if not current_dataset_mapping: continue

            methods_to_process_for_model = []
            file_name_template = "" # Will be fully formed or a template string

            if model_abbr == 'CoDE':
                methods_to_process_for_model = [CODE_RECALL_METHOD_KEY]
                # Updated filename structure for CoDE .jsonl files based on user feedback
                file_name_template = f"AI_dev-{dataset_name}-CoDE-rationales.jsonl"
            else: # LLMs
                model_full = MODEL_NAME_MAP_FULL.get(model_abbr)
                if not model_full:
                    print(f"Warning: No full model name found for LLM '{model_abbr}'. Skipping recall processing for this LLM in {dataset_name}.", file=sys.stderr)
                    continue
                methods_to_process_for_model = LLM_METHODS
                # Filename template for LLMs, method part will be filled in loop
                file_name_template = f"AI_dev-{dataset_name}-{model_full}-{{method}}-n1-wait0-rationales.jsonl"

            for method_key in methods_to_process_for_model:
                if model_abbr == 'CoDE':
                    # For CoDE, file_name_template is already the complete filename
                    file_name = file_name_template 
                else: # LLM, format the template with the specific method
                    file_name = file_name_template.format(method=method_key)
                
                file_path = os.path.join(RESPONSES_DIR, file_name)

                if not os.path.exists(file_path):
                    print(f"  Info: Recall file not found for {model_abbr} ({method_key if model_abbr != 'CoDE' else 'recall_data'}) in {dataset_name}: {file_path}")
                    recall_files_missing += 1
                    continue 

                try:
                    with open(file_path, 'r') as f:
                        try:
                            response_data = json.load(f)
                            if not isinstance(response_data, list):
                                print(f"Warning: Expected JSON list in {file_path}, got {type(response_data)}. Skipping.", file=sys.stderr)
                                recall_format_errors += 1
                                continue
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in {file_path}: {e}. Skipping.", file=sys.stderr)
                            recall_format_errors += 1
                            continue

                        for item in response_data:
                            if not isinstance(item, dict):
                                print(f"Warning: Expected dict item in {file_path}, got {type(item)}. Skipping item.", file=sys.stderr)
                                recall_errors_parsing += 1
                                continue

                            image_key = item.get('image')
                            score_val = item.get('cur_score')

                            if image_key is None or score_val is None:
                                recall_errors_parsing += 1
                                continue

                            true_subset_lower = current_dataset_mapping.get(image_key)
                            if true_subset_lower is None:
                                # print(f"Debug: Image key '{image_key}' not found in mapping for {dataset_name}. Skipping item.", file=sys.stderr)
                                recall_errors_parsing += 1
                                continue

                            try:
                                if int(score_val) == 1:
                                    all_results[dataset_name][model_abbr][method_key]['raw_counts'][true_subset_lower]['TP'] += 1
                                else:
                                    all_results[dataset_name][model_abbr][method_key]['raw_counts'][true_subset_lower]['FN'] += 1
                            except (ValueError, TypeError):
                                print(f"Warning: Invalid score '{score_val}' for image '{image_key}' in {file_path}. Skipping.", file=sys.stderr)
                                recall_errors_parsing += 1
                                continue
                            
                            active_subsets_per_dataset[dataset_name].add(true_subset_lower)
                    recall_files_processed += 1
                    print(f"  Successfully processed: {file_path}")
                except Exception as e:
                    print(f"Error processing recall file {file_path}: {e}", file=sys.stderr)
                    recall_errors_parsing += 1

    print(f"\nRecall File Processing Summary:")
    print(f"  Files Attempted (based on model/method combinations): (approximate, depends on file existence)")
    print(f"  Files Successfully Processed: {recall_files_processed}")
    print(f"  Files Missing/Not Found: {recall_files_missing}")
    print(f"  Files with JSON Format Errors: {recall_format_errors}")
    print(f"  Items with Data/Parsing Errors: {recall_errors_parsing}")

    # --- 4. Calculate Recall Scores (for ALL models) ---
    print("\nCalculating recall scores...")
    for dataset_name in all_results:
        for model_abbr in all_results[dataset_name]:
            methods_for_model = list(all_results[dataset_name][model_abbr].keys()) # Get actual methods found
            for method_key in methods_for_model:
                if 'raw_counts' in all_results[dataset_name][model_abbr][method_key]:
                    for subset_name, counts in all_results[dataset_name][model_abbr][method_key]['raw_counts'].items():
                        tp = counts['TP']
                        fn = counts['FN']
                        recall = get_recall(tp, fn)
                        all_results[dataset_name][model_abbr][method_key]['subsets'][subset_name] = round(recall * 100, 2)

    # --- 5. Load CoDe Scores from CSV --- (SECTION REMOVED as CoDe recall now comes from .jsonl)

    # --- 6. Generate LaTeX Tables ---
    print("\nGenerating LaTeX tables...")
    final_latex_string = io.StringIO()

    final_latex_string.write("% --- Suggested LaTeX Preamble ---\n")
    final_latex_string.write("% Ensure: booktabs, multirow, graphicx, caption, amsmath\n")
    final_latex_string.write("% \\documentclass[twocolumn]{article}\n")
    final_latex_string.write("% \\usepackage[margin=1in]{geometry}\n")
    final_latex_string.write("% \\usepackage{booktabs}\n")
    final_latex_string.write("% \\usepackage{multirow}\n")
    final_latex_string.write("% \\usepackage{graphicx}\n")
    final_latex_string.write("% \\usepackage{amsmath}\n")
    final_latex_string.write("% \\usepackage{caption}\n")
    final_latex_string.write("% \\usepackage{siunitx} % Optional\n")
    final_latex_string.write("% \\begin{document}\n\n")

    sorted_datasets = sorted(list(dataset_mappings.keys()))
    
    for dataset_name in sorted_datasets:
        if dataset_name not in all_results or not all_results[dataset_name]: # Check if dataset has any model data
            print(f"Skipping table for dataset '{dataset_name}' as no results were found for any model.")
            continue

        print(f"Generating table for: {dataset_name}")
        dataset_level_results = all_results.get(dataset_name, defaultdict(lambda: defaultdict(lambda: {'subsets': defaultdict(lambda: np.nan)})))

        current_active_subsets = sorted(list(active_subsets_per_dataset.get(dataset_name, set()))) # Ensure sorted order
        
        # Define display order for subsets, putting 'real' first if it exists
        sorted_subsets_display_order = []
        if not current_active_subsets:
            print(f"  Info: No active recall subsets found for dataset '{dataset_name}'. Subset columns will be based on this empty list.", file=sys.stderr)
        else:
            # Update SUBSET_NAME_MAPPING dynamically if new raw subsets found
            for s_raw in current_active_subsets: # s_raw is the key from current_dataset_mapping
                if s_raw not in SUBSET_NAME_MAPPING:
                    s_lower = s_raw.lower() # Default to lowercase if no explicit mapping
                    # Check if lowercase version already has a mapping to avoid overwriting specific mappings like "stable_diffusion_v_1_4" -> "sd1.4"
                    if s_lower not in SUBSET_NAME_MAPPING:
                         print(f"  Info: Adding dynamic mapping for raw subset '{s_raw}' -> '{s_lower}' for display.", file=sys.stderr)
                         SUBSET_NAME_MAPPING[s_raw] = s_lower
                    # If s_lower IS in SUBSET_NAME_MAPPING, it means a specific mapping (e.g. "sd1.4") exists.
                    # We should use the raw key s_raw for lookup in all_results, but display SUBSET_NAME_MAPPING[s_raw] or SUBSET_NAME_MAPPING[s_lower]
            
            # Sort based on the display names from SUBSET_NAME_MAPPING
            mapped_names_for_sort = {s_raw: SUBSET_NAME_MAPPING.get(s_raw, s_raw.lower()) for s_raw in current_active_subsets}
            # sorted_subsets_raw_keys will hold the original keys ('real', 'image_gen0', etc.) in the desired display order
            sorted_subsets_raw_keys = sorted(current_active_subsets, key=lambda s_raw: mapped_names_for_sort[s_raw])

            if 'real' in sorted_subsets_raw_keys:
                sorted_subsets_raw_keys.remove('real')
                sorted_subsets_display_order = ['real'] + sorted_subsets_raw_keys
            else:
                sorted_subsets_display_order = sorted_subsets_raw_keys

        final_latex_string.write(f"% --- Table for Dataset: {dataset_name} ---\n")
        final_latex_string.write(r"\begin{table*}[htbp]" + "\n")
        final_latex_string.write(r"\centering" + "\n")
        dataset_display_name = escape_latex(dataset_name.upper())
        final_latex_string.write(f"\\caption{{Recall Scores for {dataset_display_name} Dataset}}\n") # Updated caption
        final_latex_string.write(f"\\label{{tab:{dataset_name}_scores}}\n")
        final_latex_string.write(r"\setlength{\tabcolsep}{3pt}" + "\n")
        final_latex_string.write(TABLE_FONT_SIZE + "\n")

        num_subset_cols = len(sorted_subsets_display_order)
        recall_col_spec = 'l' * num_subset_cols if num_subset_cols > 0 else ''
        col_spec = 'll' + recall_col_spec 
        final_latex_string.write(r"\begin{tabular}{" + col_spec + "}\n")
        final_latex_string.write(r"\toprule" + "\n")

        header_line1 = [r"\textbf{Model}", r"\textbf{Method}"] # Header reverted
        if num_subset_cols > 0:
            for subset_raw_key in sorted_subsets_display_order: # Iterate using raw keys
                # Get display name from mapping, default to raw key's lower if not found
                subset_display_name_from_map = SUBSET_NAME_MAPPING.get(subset_raw_key, subset_raw_key.lower())
                subset_display_escaped = escape_latex(subset_display_name_from_map)
                header_line1.append(rf"\textbf{{{subset_display_escaped}}}")
        
        final_latex_string.write(" & ".join(header_line1) + r" \\" + "\n")
        final_latex_string.write(r"\midrule" + "\n")

        models_in_table_for_dataset = [m for m in MODELS_ABBR if m in dataset_level_results and dataset_level_results[m]]

        for model_idx, model_abbr in enumerate(models_in_table_for_dataset):
            model_specific_data = dataset_level_results.get(model_abbr, defaultdict(lambda: {'subsets': defaultdict(lambda: np.nan)}))
            
            # Determine methods for this model (LLM_METHODS or CODE_RECALL_METHOD_KEY)
            current_model_methods = []
            if model_abbr == 'CoDE':
                if CODE_RECALL_METHOD_KEY in model_specific_data: # Check if CoDE recall data was loaded
                    current_model_methods = [CODE_RECALL_METHOD_KEY]
            else: # LLM
                current_model_methods = [m for m in LLM_METHODS if m in model_specific_data] # Only methods with data

            if not current_model_methods: # Skip if model has no methods with data
                print(f"  Skipping {model_abbr} in {dataset_name} table as no method data was found after processing.")
                continue

            # Max recall calculation for this model (LLM or CoDE)
            max_recall_per_subset = defaultdict(lambda: -1.0)
            if BOLD_MAX_RECALL_PER_MODEL and num_subset_cols > 0: # This outer BOLD_MAX_RECALL_PER_MODEL check is global
                for method_k in current_model_methods:
                    method_data_for_max = model_specific_data.get(method_k, {})
                    for subset_raw_k in sorted_subsets_display_order: # Use raw key for data lookup
                        recall_val = method_data_for_max.get('subsets', {}).get(subset_raw_k, np.nan)
                        if pd.notna(recall_val):
                            max_recall_per_subset[subset_raw_k] = max(max_recall_per_subset[subset_raw_k], recall_val)
            
            # Iterate through the model's methods
            for i, method_key_for_row in enumerate(current_model_methods):
                row_cells = []
                model_display_name_escaped = escape_latex(model_abbr)

                if len(current_model_methods) == 1: # Single row for this model (e.g., CoDE or LLM with only one method's data)
                    row_cells.append(model_display_name_escaped)
                elif i == 0: # First method for this multi-method LLM
                    row_cells.append(f"\\multirow{{{len(current_model_methods)}}}{{*}}{{{model_display_name_escaped}}}")
                else: # Subsequent methods for this LLM
                    row_cells.append("")

                # Method display name
                method_display_str = ""
                if model_abbr == 'CoDE' and method_key_for_row == CODE_RECALL_METHOD_KEY:
                    method_display_str = CODE_DISPLAY_METHOD_NAME # e.g., "Recall"
                else: # LLM
                    method_display_raw = METHOD_NAME_MAPPING.get(method_key_for_row, method_key_for_row)
                    method_display_str = method_display_raw # Already LaTeX formatted if needed
                    if method_key_for_row == METHOD_TO_ADD_OURS_SUFFIX:
                        method_display_str += f"\\ {DIFF_FONT_SIZE}{{(ours)}}"
                row_cells.append(method_display_str)
                
                current_method_data = model_specific_data.get(method_key_for_row, {'subsets': {}})
                subset_scores_for_method = current_method_data.get('subsets', {})

                if num_subset_cols > 0:
                    for subset_raw_k_for_cell in sorted_subsets_display_order: # Use raw key for data lookup
                        recall_val = subset_scores_for_method.get(subset_raw_k_for_cell, np.nan)
                        formatted_recall_val = format_score(recall_val) 
                        
                        current_max_recall_for_subset = max_recall_per_subset.get(subset_raw_k_for_cell, -1.0)
                        # MODIFIED: Added "and model_abbr != 'CoDE'" to prevent bolding for CoDE
                        if BOLD_MAX_RECALL_PER_MODEL and model_abbr != 'CoDE' and \
                           pd.notna(recall_val) and np.isclose(recall_val, current_max_recall_for_subset):
                            formatted_recall_val = f"\\textbf{{{formatted_recall_val}}}"
                        row_cells.append(formatted_recall_val)
                
                final_latex_string.write(" & ".join(row_cells) + r" \\" + "\n")

            if model_idx < len(models_in_table_for_dataset) - 1:
                final_latex_string.write(r"\midrule" + "\n")

        final_latex_string.write(r"\bottomrule" + "\n")
        final_latex_string.write(r"\end{tabular}" + "\n")
        final_latex_string.write(r"\end{table*}" + "\n\n")

    final_latex_string.write("% --- End of Generated Tables ---\n")

    # --- 7. Output ---
    final_output = final_latex_string.getvalue()
    final_latex_string.close()

    print("\n" + "="*25 + " Generated LaTeX Code " + "="*25)
    print(final_output)
    print("="*70)
    print("\nNotes (Updated):")
    print("- CoDe recall scores are now calculated from .jsonl files in RESPONSES_DIR.")
    print("-   (Updated CoDe .jsonl filename assumption: AI_dev-{dataset}-CoDE-rationales.jsonl)")
    print("- Removed loading of CoDe scores from CSV files.")
    print("- All models listed in MODELS_ABBR are processed for recall from .jsonl files.")
    print("- Table caption and 'Method' header updated for general recall scores.")
    print("- CoDe is displayed with 'Recall' as its method and its per-subset recall scores.")
    print("- Recall values for 'CoDE' model will not be bolded, even if BOLD_MAX_RECALL_PER_MODEL is True.")


if __name__ == "__main__":
    main()
