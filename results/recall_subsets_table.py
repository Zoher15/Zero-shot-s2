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
MODELS_ABBR = ['qwen2.5', 'llama3.2','CoDE']
MODEL_NAME_MAP_FULL = {'llama3.2': 'llama3-11b', 'qwen2.5': 'qwen25-7b'} # For LLM .jsonl filenames

# LLM_METHODS applies to LLMs. CoDE will have its own handling for its .jsonl.
LLM_METHODS = ['zeroshot', 'zeroshot-cot', 'zeroshot-2-artifacts']
DATASETS = ['d3', 'df40', 'genimage'] # Ensure these match the keys in font size dicts

# Subset Naming
SUBSET_NAME_MAPPING = {
    "real": "real", "image_gen0": "dfif", "image_gen1": "sd1.4", "image_gen2": "sd2.1", "image_gen3": "sdxl",
    "collabdiff": "cdif", "midjourney": "midj", "stargan": "sgan1", "starganv2": "sgan2", "styleclip": "sclip",
    "whichfaceisreal": "wfir", "adm": "adm", "biggan": "bgan", "glide_copy": "glide",
    "stable_diffusion_v_1_4": "sd1.4", "stable_diffusion_v_1_5": "sd1.5", "vqdm": "vqdm", "wukong": "wk"
}

# Method Name Mapping (for LLMs)
METHOD_NAME_MAPPING = {
    'zeroshot': 'zero-shot',
    'zeroshot-cot': 'zero-shot-cot',
    'zeroshot-2-artifacts': 'zero-shot-s$^2$' # Display name for zeroshot-2-artifacts
}
METHOD_TO_ADD_OURS_SUFFIX = 'zeroshot-2-artifacts'
# Only 'zeroshot-2-artifacts' (s2) will show diff
METHODS_TO_SHOW_DIFF = ['zeroshot-2-artifacts']
# Baseline for the diff is now 'zeroshot-cot'
BASELINE_METHOD_FOR_DIFF = 'zeroshot-cot'

# LaTeX Formatting Options
BOLD_MAX_RECALL_PER_MODEL = True # If True, bold the max recall score per model (excluding CoDe) for each subset
ZERO_PAD_RECALL = True # If True, pad single-digit recall scores with a leading zero (e.g., 07.1)

# --- Dataset-Specific Font Sizes ---
# Define fallback defaults in case a dataset is processed that isn't listed below
DEFAULT_TABLE_FONT_SIZE = r"\footnotesize"
DEFAULT_DIFF_FONT_SIZE = r"\fontsize{5}{4}\selectfont"

# Dictionary for dataset-specific table font sizes
# Keys should match the dataset names in the DATASETS list
DATASET_TABLE_FONT_SIZES = {
    'd3':       r"\small", # Example size for d3
    'df40':     r"\footnotesize", # Example size for df40
    'genimage': r"\scriptsize",       # Smaller font for genimage
}
# Dictionary for dataset-specific difference font sizes
# Keys should match the dataset names in the DATASETS list
DATASET_DIFF_FONT_SIZES = {
    'd3':       r"\tiny",   # Example size for d3 differences
    'df40':     r"\tiny",   # Example size for df40 differences
    'genimage': r"\fontsize{5}{5}\selectfont", # Smaller font size for genimage differences
}


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
                image_path = os.path.join(file_dir, file) # Keep full path if needed later, or just filename
                base_name = file.replace('.png', '')
                parts = base_name.split('_')
                raw_subset = 'unknown'

                # D3 Naming convention check:
                # Examples: real_ILSVRC2012_val_00000001.png -> real
                #           biggan_0.png -> biggan
                #           stable_diffusion_v_1_4_100.png -> stable_diffusion_v_1_4
                if len(parts) > 1:
                    if 'real' in parts: # Check if 'real' is explicitly mentioned
                        raw_subset = 'real'
                    else:
                        # Attempt to join parts excluding potential numeric suffix
                        joined_parts = "_".join(parts[1:]) # Join everything after the first part (often 'fake')
                        raw_subset_candidate = re.sub(r'_\d+$', '', joined_parts) # Remove trailing _<number>
                        if raw_subset_candidate:
                             raw_subset = raw_subset_candidate
                        else: # Fallback if stripping digits leaves nothing or structure is different
                             if len(parts) > 2: # e.g., fake_biggan_0 -> take biggan
                                 raw_subset = "_".join(parts[1:-1])
                             else: # e.g., fake_biggan -> take biggan? Needs clarification. Assume parts[1] is subset for now.
                                 raw_subset = parts[1] if len(parts) > 1 else 'unknown'
                                 print(f"Warning: D3 file '{file}' - Ambiguous subset pattern after joining/stripping: {parts}. Using '{raw_subset}'.", file=sys.stderr)

                else: # Single part filename (e.g., 'real.png' - unlikely based on examples)
                    raw_subset = parts[0] if parts else 'unknown'
                    print(f"Warning: Could not determine subset pattern for D3 file (too few parts): {file}. Using '{raw_subset}'.", file=sys.stderr)

                subset = str(raw_subset).lower()
                # Use the filename as the key, assuming responses also use filename
                subset_data[file] = subset # Keying by filename `file` instead of `image_path`
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
            image = row['file_path'] # This seems to be the relative path used as key
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
    # Order matters: escape backslash first
    text = text.replace('\\', r'\textbackslash{}')
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('#', r'\#')
    text = text.replace('_', r'\_')
    text = text.replace('{', r'\{')
    text = text.replace('}', r'\}')
    text = text.replace('~', r'\textasciitilde{}')
    # Do not escape $ and ^ as they are used in method names like s$^2$
    return text

def format_score(score, diff=None, diff_font_size_str=None):
    """
    Formats recall score for LaTeX table with optional zero padding to 1 decimal place.
    Optionally includes a difference value in parentheses with a specific font size string.
    Returns the formatted string. Bolding is handled separately.

    Args:
        score: The recall score (float or convertible to float).
        diff: The difference value (float or convertible to float), or None.
        diff_font_size_str: The LaTeX command string for the difference font size (e.g., r"\fontsize{5}{4}\selectfont").
    """
    if pd.isna(score) or score is None:
        return '-'
    zero_pad = ZERO_PAD_RECALL
    try:
        score_float = float(score)
    except ValueError:
        return '-' # Return '-' if score is not a valid number

    # Format the base score value to 1 decimal place
    formatted_val_base = f"{score_float:.1f}"

    # Apply zero padding if needed
    if zero_pad and score_float >= 0 and score_float < 10 and not formatted_val_base.startswith('0.'):
         # Check if it's an integer like 7.0, 8.0 etc. before padding
        if '.' in formatted_val_base and formatted_val_base.endswith('.0'):
             formatted_val_base = f"0{formatted_val_base}"
        elif '.' not in formatted_val_base: # Handle integer case if formatting resulted in '7' instead of '7.0'
             formatted_val_base = f"0{score_float:.1f}" # Reformat to ensure .0

    # Add the difference part if provided, valid, and a font size string is given
    if diff is not None and pd.notna(diff) and diff_font_size_str:
        try:
            diff_float = float(diff)
            formatted_diff = f"{diff_float:+.1f}" # Format diff with sign and 1 decimal place
            # Append the difference string with its specific font size - REMOVED leading space
            formatted_val_base += f"{{\\ {diff_font_size_str}({formatted_diff})}}"
        except ValueError:
            pass # Ignore invalid diff values

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
    # Filter out datasets that failed to load or are not in the requested list
    dataset_mappings = {k: v for k, v in dataset_mappings.items() if v and k in DATASETS}
    if not dataset_mappings:
        print("Error: No dataset mappings loaded successfully. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"\nLoaded mappings for datasets: {list(dataset_mappings.keys())}")

    # --- 2. Initialize Data Structures ---
    # Structure: all_results[dataset][model][method]['subsets'][subset_name] = recall_score
    # Structure: all_results[dataset][model][method]['raw_counts'][subset_name] = {'TP': count, 'FN': count}
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'subsets': defaultdict(lambda: np.nan), 'raw_counts': defaultdict(lambda: {'TP': 0, 'FN': 0})})))
    # Keep track of subsets actually encountered in results for each dataset
    active_subsets_per_dataset = defaultdict(set)

    # --- 3. Process Recall Data from Response Files ---
    print("\nProcessing .jsonl response files for recall calculation...")
    recall_files_processed = 0
    recall_files_missing = 0
    recall_errors_parsing = 0 # Count of items within files that couldn't be parsed
    recall_format_errors = 0 # Count of files with major format issues (not JSON list, decode error)

    # Iterate through configured models, datasets, and methods
    for model_abbr in MODELS_ABBR:
        for dataset_name in dataset_mappings.keys():
            current_dataset_mapping = dataset_mappings[dataset_name]
            if not current_dataset_mapping: continue # Skip if mapping failed for this dataset

            methods_to_process_for_model = []
            file_name_template = ""

            # Determine methods and filename template based on model type
            if model_abbr == 'CoDE':
                methods_to_process_for_model = [CODE_RECALL_METHOD_KEY] # Use internal key
                # CoDE filename pattern (adjust if needed)
                file_name_template = f"AI_dev-{dataset_name}-CoDE-rationales.jsonl"
            else: # Handle LLMs
                model_full = MODEL_NAME_MAP_FULL.get(model_abbr)
                if not model_full:
                    print(f"Warning: No full model name found for LLM '{model_abbr}'. Skipping for {dataset_name}.", file=sys.stderr)
                    continue
                methods_to_process_for_model = LLM_METHODS

                # Determine filename prefix based on model name convention
                filename_prefix = "AI_dev" # Default prefix
                if "llama" in model_abbr.lower():
                    filename_prefix = "AI_util" # Specific prefix for llama models

                # LLM filename pattern
                file_name_template = f"{filename_prefix}-{dataset_name}-{model_full}-{{method}}-n1-wait0-rationales.jsonl"

            # Process each method for the current model and dataset
            for method_key in methods_to_process_for_model:
                if model_abbr == 'CoDE':
                    file_name = file_name_template # CoDE template doesn't need method formatting
                else:
                    file_name = file_name_template.format(method=method_key) # Format LLM filename

                file_path = os.path.join(RESPONSES_DIR, file_name)

                if not os.path.exists(file_path):
                    print(f"   Info: Recall file not found: {file_path}")
                    recall_files_missing += 1
                    continue # Skip to the next method/file

                # Try reading and processing the JSONL file
                try:
                    with open(file_path, 'r') as f:
                        try:
                            # Load the entire JSON structure (assuming it's a list of dicts)
                            response_data = json.load(f)
                            if not isinstance(response_data, list):
                                print(f"Warning: Expected JSON list in {file_path}, got {type(response_data)}. Skipping.", file=sys.stderr)
                                recall_format_errors += 1
                                continue
                        except json.JSONDecodeError as e:
                                print(f"Error decoding JSON in {file_path}: {e}. Skipping.", file=sys.stderr)
                                recall_format_errors += 1
                                continue

                        items_processed_in_file = 0
                        # Iterate through each item (presumably a dictionary) in the loaded list
                        for item in response_data:
                            if not isinstance(item, dict):
                                print(f"Warning: Expected dict item in {file_path}, got {type(item)}. Skipping item.", file=sys.stderr)
                                recall_errors_parsing += 1
                                continue

                            # Extract image identifier and score
                            # Adjust 'image_key_in_json' if the key is different in the JSON files
                            image_key_in_json = 'image' # Assumed key for the image identifier
                            image_val = item.get(image_key_in_json)
                            score_val = item.get('cur_score') # Assumed key for the score (0 or 1)

                            if image_val is None or score_val is None:
                                print(f"Warning: Missing '{image_key_in_json}' or 'cur_score' in item: {item} in {file_path}. Skipping item.", file=sys.stderr)
                                recall_errors_parsing += 1
                                continue

                            # --- Image Key Matching Logic ---
                            # Determine the correct key to use for lookup in the dataset mapping
                            lookup_key = image_val # Default assumption
                            if dataset_name == 'd3':
                                # D3 mapping uses filename, extract filename from path if needed
                                lookup_key = os.path.basename(str(image_val))
                            elif dataset_name == 'df40' or dataset_name == 'genimage':
                                # DF40 and GenImage mappings use the path as provided in their CSVs
                                lookup_key = str(image_val) # Ensure it's a string

                            # Find the true subset using the appropriate key
                            true_subset_lower = current_dataset_mapping.get(lookup_key)

                            if true_subset_lower is None:
                                # print(f"Warning: Image key '{lookup_key}' (from value '{image_val}') not found in {dataset_name} mapping. Skipping item.", file=sys.stderr)
                                recall_errors_parsing += 1
                                continue # Skip if image key not in our loaded mapping

                            # Tally True Positives (TP) and False Negatives (FN)
                            try:
                                if int(score_val) == 1: # Assuming 1 means correct detection (TP for 'fake' class recall)
                                    all_results[dataset_name][model_abbr][method_key]['raw_counts'][true_subset_lower]['TP'] += 1
                                else: # Assuming 0 means missed detection (FN)
                                    all_results[dataset_name][model_abbr][method_key]['raw_counts'][true_subset_lower]['FN'] += 1
                            except (ValueError, TypeError):
                                print(f"Warning: Invalid score value '{score_val}' for key '{lookup_key}' in {file_path}. Skipping item.", file=sys.stderr)
                                recall_errors_parsing += 1
                                continue

                            # Record that this subset was encountered for this dataset
                            active_subsets_per_dataset[dataset_name].add(true_subset_lower)
                            items_processed_in_file += 1

                        # Report file processing status
                        if items_processed_in_file > 0:
                            recall_files_processed += 1
                            print(f"   Successfully processed {items_processed_in_file} items from: {file_path}")
                        elif isinstance(response_data, list) and not response_data: # File was empty list
                             print(f"   Info: Processed empty list from: {file_path}")
                             recall_files_processed +=1 # Count as processed even if empty
                        else: # File had format errors or no processable items
                             print(f"   Warning: No valid items processed in: {file_path}")

                except Exception as e: # Catch errors during file opening or general processing
                    print(f"Error during processing of file {file_path}: {e}", file=sys.stderr)
                    # Decide if this counts as a format error or just skip
                    # recall_format_errors += 1 # Optionally count this as a format error

    # Print summary of file processing
    print(f"\nRecall File Processing Summary:")
    print(f"   Files Successfully Processed (or empty): {recall_files_processed}")
    print(f"   Files Missing/Not Found: {recall_files_missing}")
    print(f"   Files with JSON Format/Decode Errors: {recall_format_errors}")
    print(f"   Items with Data/Parsing Errors (within files): {recall_errors_parsing}")


    # --- 4. Calculate Recall Scores ---
    print("\nCalculating recall scores...")
    for dataset_name in all_results:
        for model_abbr in all_results[dataset_name]:
            methods_for_model = list(all_results[dataset_name][model_abbr].keys()) # Get methods processed for this model/dataset
            for method_key in methods_for_model:
                # Check if raw counts exist before calculating recall
                if 'raw_counts' in all_results[dataset_name][model_abbr][method_key]:
                    for subset_name, counts in all_results[dataset_name][model_abbr][method_key]['raw_counts'].items():
                        tp = counts['TP']
                        fn = counts['FN']
                        recall = get_recall(tp, fn)
                        # Store recall score, rounded to 1 decimal place
                        all_results[dataset_name][model_abbr][method_key]['subsets'][subset_name] = round(recall * 100, 1)


    # --- 5. Generate LaTeX Tables ---
    print("\nGenerating LaTeX tables...")
    final_latex_string = io.StringIO() # Use StringIO to build the LaTeX string in memory

    # Add suggested LaTeX preamble comments
    final_latex_string.write("% --- Suggested LaTeX Preamble ---\n")
    final_latex_string.write("% \\usepackage{booktabs, multirow, graphicx, caption, amsmath, siunitx, amsfonts}\n")
    final_latex_string.write("% \\documentclass[twocolumn]{article} % Or your document class\n")
    final_latex_string.write("% \\begin{document}\n\n")

    # Sort datasets alphabetically for consistent table order
    sorted_datasets = sorted(list(dataset_mappings.keys()))

    for dataset_name in sorted_datasets:
        # Check if there are any results for this dataset before creating a table
        if dataset_name not in all_results or not all_results[dataset_name]:
            print(f"Skipping table for dataset '{dataset_name}' as no results were found.")
            continue

        print(f"Generating table for: {dataset_name}")
        # Get results specific to this dataset
        dataset_level_results = all_results.get(dataset_name, defaultdict(lambda: defaultdict(lambda: {'subsets': defaultdict(lambda: np.nan)})))
        # Get the active subsets encountered for this dataset and sort them
        current_active_subsets_raw = sorted(list(active_subsets_per_dataset.get(dataset_name, set())))

        # --- Subset Name Mapping and Sorting ---
        sorted_subsets_display_order_raw = [] # List to hold the raw subset keys in desired display order
        if not current_active_subsets_raw:
            print(f"   Info: No active subsets found for '{dataset_name}'. Table columns might be empty.", file=sys.stderr)
        else:
            # Ensure all active raw subsets have a mapping (default to lowercase if missing)
            for s_raw in current_active_subsets_raw:
                if s_raw not in SUBSET_NAME_MAPPING:
                    s_lower = s_raw.lower()
                    # Check if the lowercase version is already a *value* in the mapping
                    if s_lower not in SUBSET_NAME_MAPPING.values():
                         # If neither raw nor lower is key/value, add mapping raw -> lower
                         SUBSET_NAME_MAPPING[s_raw] = s_lower
                         print(f"   Info: Added default mapping for subset '{s_raw}' -> '{s_lower}'")
                    # If lower is already a value, we need to map s_raw to it, but avoid overwriting if s_raw is already a key
                    elif SUBSET_NAME_MAPPING.get(s_raw) is None:
                         SUBSET_NAME_MAPPING[s_raw] = s_lower
                         # print(f"   Info: Mapped '{s_raw}' to existing value '{s_lower}'")


            # Create a dictionary for sorting based on the mapped display names
            mapped_names_for_sort = {s_raw: SUBSET_NAME_MAPPING.get(s_raw, s_raw.lower()) for s_raw in current_active_subsets_raw}
            # Sort the raw keys based on their mapped display names
            sorted_subsets_raw_keys_temp = sorted(current_active_subsets_raw, key=lambda s_raw: mapped_names_for_sort[s_raw])

            # Move 'real' subset to the beginning if it exists
            if 'real' in sorted_subsets_raw_keys_temp:
                sorted_subsets_raw_keys_temp.remove('real')
                sorted_subsets_display_order_raw = ['real'] + sorted_subsets_raw_keys_temp
            else:
                sorted_subsets_display_order_raw = sorted_subsets_raw_keys_temp


        # --- Start LaTeX Table Environment ---
        final_latex_string.write(f"% --- Table for Dataset: {dataset_name} ---\n")
        final_latex_string.write(r"\begin{table*}[htbp]" + "\n") # Use table* for full width
        final_latex_string.write(r"\centering" + "\n")

        # Create caption and label
        dataset_display_name = escape_latex(dataset_name.upper())
        baseline_method_display = escape_latex(METHOD_NAME_MAPPING.get(BASELINE_METHOD_FOR_DIFF, BASELINE_METHOD_FOR_DIFF))
        method_s2_display = escape_latex(METHOD_NAME_MAPPING.get('zeroshot-2-artifacts','zeroshot-s$^2$')) # Use mapped name
        # Updated caption: Removed specific font size mention for difference
        caption_text = f"Recall Scores for {dataset_display_name}. Scores to 1 d.p. Differences for {method_s2_display} vs {baseline_method_display} shown in parentheses."
        final_latex_string.write(f"\\caption{{{caption_text}}}\n")
        final_latex_string.write(f"\\label{{tab:{dataset_name}_scores}}\n")

        # Set table properties (column separation, font size)
        final_latex_string.write(r"\setlength{\tabcolsep}{3pt}" + "\n") # Adjust column spacing
        # Get the appropriate font size for this dataset, using fallback if key missing
        current_table_font_size = DATASET_TABLE_FONT_SIZES.get(dataset_name, DEFAULT_TABLE_FONT_SIZE)
        final_latex_string.write(current_table_font_size + "\n") # Apply table font size
        # Get the appropriate difference font size for this dataset, using fallback if key missing
        current_diff_font_size = DATASET_DIFF_FONT_SIZES.get(dataset_name, DEFAULT_DIFF_FONT_SIZE)


        # Define column specification: Model (l), Method (l), one 'l' for each subset
        num_subset_cols = len(sorted_subsets_display_order_raw)
        recall_col_spec = 'l' * num_subset_cols if num_subset_cols > 0 else '' # Left-align recall columns
        col_spec = 'll' + recall_col_spec # ll for Model and Method columns
        final_latex_string.write(r"\begin{tabular}{" + col_spec + "}\n")
        final_latex_string.write(r"\toprule" + "\n") # Top rule

        # --- Table Header ---
        header_line1 = [r"\textbf{Model}", r"\textbf{Method}"]
        if num_subset_cols > 0:
            for subset_raw_key in sorted_subsets_display_order_raw:
                # Get the display name from mapping, default to raw key lowercased
                subset_display_name_from_map = SUBSET_NAME_MAPPING.get(subset_raw_key, subset_raw_key.lower())
                subset_display_escaped = escape_latex(subset_display_name_from_map)
                header_line1.append(rf"\textbf{{{subset_display_escaped}}}") # Add bold subset name
        else:
             # Add a placeholder if no subsets, though table might look odd
             # header_line1.append(r"\textbf{Recall}")
             pass # Or just have Model/Method columns

        final_latex_string.write(" & ".join(header_line1) + r" \\" + "\n") # Write header row
        final_latex_string.write(r"\midrule" + "\n") # Mid rule after header

        # --- Table Body ---
        # Get models that actually have results for this dataset
        models_in_table_for_dataset = [m for m in MODELS_ABBR if m in dataset_level_results and dataset_level_results[m]]

        for model_idx, model_abbr in enumerate(models_in_table_for_dataset):
            model_specific_data = dataset_level_results.get(model_abbr, defaultdict(lambda: {'subsets': defaultdict(lambda: np.nan)}))

            # Determine which methods have data for this model
            current_model_methods_with_data = []
            if model_abbr == 'CoDE':
                if CODE_RECALL_METHOD_KEY in model_specific_data:
                    current_model_methods_with_data = [CODE_RECALL_METHOD_KEY]
            else: # LLMs
                # Use the defined order of LLM methods
                ordered_llm_methods = [m for m in LLM_METHODS if m in model_specific_data]
                current_model_methods_with_data = ordered_llm_methods

            if not current_model_methods_with_data:
                print(f"   Skipping {model_abbr} in {dataset_name} table as no method data was found.")
                continue # Skip this model if no methods have data

            # --- Calculate Max Recall and Baseline Scores (for bolding/diff) ---
            max_recall_per_subset = defaultdict(lambda: -1.0) # Store max recall per subset for this model
            baseline_scores = {} # Store baseline scores for calculating differences

            # Find max recall per subset *within this model* (only for LLMs, not CoDe)
            if BOLD_MAX_RECALL_PER_MODEL and num_subset_cols > 0 and model_abbr != 'CoDE':
                for method_k in current_model_methods_with_data:
                    method_data_for_max = model_specific_data.get(method_k, {})
                    subset_scores_for_max = method_data_for_max.get('subsets', {})
                    for subset_raw_k in sorted_subsets_display_order_raw:
                        recall_val = subset_scores_for_max.get(subset_raw_k, np.nan)
                        if pd.notna(recall_val):
                            # Update max if current recall is strictly greater
                            # Use np.isclose for float comparison robustness if needed, but > should be fine here
                            if recall_val > max_recall_per_subset[subset_raw_k]:
                                max_recall_per_subset[subset_raw_k] = recall_val

            # Get baseline scores if applicable (only for LLMs and if baseline method exists)
            if model_abbr != 'CoDE' and BASELINE_METHOD_FOR_DIFF in current_model_methods_with_data:
                baseline_method_data = model_specific_data.get(BASELINE_METHOD_FOR_DIFF, {})
                baseline_scores = baseline_method_data.get('subsets', {})


            # --- Generate Rows for Each Method ---
            num_methods_for_multirow = len(current_model_methods_with_data)
            for i, method_key_for_row in enumerate(current_model_methods_with_data):
                row_cells = [] # Start collecting cells for this row

                # --- Model Name Cell (with multirow if needed) ---
                model_display_name_escaped = escape_latex(model_abbr)
                if num_methods_for_multirow == 1: # Only one method for this model
                    row_cells.append(model_display_name_escaped)
                elif i == 0: # First row for this model
                    # Use multirow spanning all method rows for this model
                    row_cells.append(f"\\multirow{{{num_methods_for_multirow}}}{{*}}{{{model_display_name_escaped}}}")
                else: # Subsequent rows for the same model
                    row_cells.append("") # Empty cell, handled by multirow

                # --- Method Name Cell ---
                method_display_str = ""
                if model_abbr == 'CoDE' and method_key_for_row == CODE_RECALL_METHOD_KEY:
                    method_display_str = escape_latex(CODE_DISPLAY_METHOD_NAME) # Use specific display name for CoDe
                else: # LLMs
                    # Get display name from mapping, default to the key itself
                    method_display_raw = METHOD_NAME_MAPPING.get(method_key_for_row, method_key_for_row)
                    # Escape the display name (handles s$^2$)
                    method_display_str = escape_latex(method_display_raw) # Already escaped where needed

                row_cells.append(method_display_str)

                # --- Recall Score Cells ---
                current_method_data = model_specific_data.get(method_key_for_row, {'subsets': {}})
                subset_scores_for_method = current_method_data.get('subsets', {})

                if num_subset_cols > 0:
                    for subset_raw_k_for_cell in sorted_subsets_display_order_raw:
                        recall_val = subset_scores_for_method.get(subset_raw_k_for_cell, np.nan)
                        diff_val = np.nan # Initialize difference as NaN

                        # Calculate difference if this method should show it
                        if model_abbr != 'CoDE' and method_key_for_row in METHODS_TO_SHOW_DIFF:
                            baseline_recall_for_diff = baseline_scores.get(subset_raw_k_for_cell, np.nan)
                            # Only calculate diff if both current and baseline scores are valid numbers
                            if pd.notna(recall_val) and pd.notna(baseline_recall_for_diff):
                                diff_val = recall_val - baseline_recall_for_diff

                        # Get the fully formatted value (score + potentially diff) using format_score
                        # Pass the dataset-specific difference font size string
                        formatted_recall_val = format_score(
                            recall_val,
                            diff=diff_val if pd.notna(diff_val) else None,
                            diff_font_size_str=current_diff_font_size
                        )

                        # --- Apply Bolding (if enabled and applicable) ---
                        # Bolding applies only to LLMs, if the score is valid and matches the max for this model/subset
                        current_max_recall_for_subset = max_recall_per_subset.get(subset_raw_k_for_cell, -1.0)
                        if BOLD_MAX_RECALL_PER_MODEL and model_abbr != 'CoDE' and pd.notna(recall_val) and np.isclose(recall_val, current_max_recall_for_subset):
                             # Check if the formatted value contains the difference part
                             # Use the current_diff_font_size for the marker
                             # Marker now starts directly with {{\\ without the leading space
                             diff_start_marker = f"{{\\ {current_diff_font_size}("

                             if diff_start_marker in formatted_recall_val:
                                 # Split the string at the start of the difference part
                                 parts = formatted_recall_val.split(diff_start_marker, 1)
                                 score_part = parts[0] # The score number itself
                                 # Reconstruct the difference part including the marker
                                 diff_part = diff_start_marker + parts[1]
                                 # Apply bold only to the score part and append the unbolded diff part
                                 formatted_recall_val = f"\\textbf{{{score_part}}}{diff_part}"
                             else:
                                 # No difference part found, safe to bold the whole score string
                                 formatted_recall_val = f"\\textbf{{{formatted_recall_val}}}"
                        # --- End Bolding Logic ---

                        row_cells.append(formatted_recall_val) # Add the (potentially bolded) cell value
                else:
                     # Handle case with no subset columns (maybe add a placeholder '-')
                     # row_cells.append("-")
                     pass

                # Write the completed row to the LaTeX string
                final_latex_string.write(" & ".join(row_cells) + r" \\" + "\n")

            # Add a midrule between different models (unless it's the last model)
            if model_idx < len(models_in_table_for_dataset) - 1:
                final_latex_string.write(r"\midrule" + "\n")

        # --- End Table Body ---

        final_latex_string.write(r"\bottomrule" + "\n") # Bottom rule
        final_latex_string.write(r"\end{tabular}" + "\n")
        final_latex_string.write(r"\end{table*}" + "\n\n") # End table* environment

    # Add closing comments for LaTeX document structure
    final_latex_string.write("% --- End of Generated Tables ---\n")
    final_latex_string.write("% \\end{document}\n")

    # --- 7. Output ---
    final_output = final_latex_string.getvalue() # Get the complete LaTeX string
    final_latex_string.close() # Close the StringIO object

    # Print the generated LaTeX code to the console
    print("\n" + "="*25 + " Generated LaTeX Code " + "="*25)
    print(final_output)
    print("="*70)

    # Print summary notes
    print("\nNotes:")
    print(f"- Bolding ({BOLD_MAX_RECALL_PER_MODEL=}) applies only to max LLM scores per subset, not CoDe or difference values.")
    print("- Table font sizes are defined per dataset in DATASET_TABLE_FONT_SIZES.")
    print("- Difference font sizes are defined per dataset in DATASET_DIFF_FONT_SIZES.")
    print(f"- Fallback table font size: {DEFAULT_TABLE_FONT_SIZE}. Fallback difference font size: {DEFAULT_DIFF_FONT_SIZE}.")
    print(f"- Differences shown for '{method_s2_display}' relative to '{baseline_method_display}'.")
    print(f"- Recall scores/differences formatted to 1 decimal place ({ZERO_PAD_RECALL=}).")
    print("- JSON files read using `json.load(f)` assuming a list structure.")
    print("- CoDe recall scores calculated from its .jsonl file.")
    print("- LLM .jsonl filenames use 'AI_util' for 'llama', 'AI_dev' for others.")
    print("- Image key matching adjusted for D3 (uses filename) vs DF40/GenImage (uses path).")
    print("- Space between score and difference value has been removed.")


if __name__ == "__main__":
    main()
