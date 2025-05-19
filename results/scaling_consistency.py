import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # Imported for Y-axis tick formatting
from sklearn.metrics import f1_score
import sys

# --- Configuration ---
# IMPORTANT: Update this path to your actual responses directory
RESPONSES_DIR = '/data3/zkachwal/visual-reasoning/data/ai-generation/responses/'
# IMPORTANT: Update this to the exact model name string used in your filenames for Llama 3
TARGET_LLAMA_MODEL_NAME = "llama3-11b" 

# Directory to store/load F1 score results for this plotting script
F1_PLOT_CACHE_DIR = "./f1_cache_plot/"

# N values to plot on the x-axis (for data fetching and ticks)
N_VALUES = [1, 5, 10, 20]
# Wait parameter (assuming it's fixed for these experiments)
WAIT_VAL_PLOT = "0"

# Datasets to create subplots for (internal_name: display_name)
DATASETS_TO_PLOT = {
    "d32k": "D3 (2k)",
    "df402k": "DF40 (2k)",
    "genimage2k": "GenImage (2k)"
}

# Methods to plot as lines (internal keys)
METHODS_TO_PLOT = ["zeroshot", "zeroshot-cot", "zeroshot-2-artifacts"]

# --- PLEASE UPDATE THIS MAPPING with your desired display names for the legend ---
METHOD_DISPLAY_NAME_MAP = {
    "zeroshot": "zero-shot", 
    "zeroshot-cot": "zero-shot-cot", 
    "zeroshot-2-artifacts": r"zero-shot-s$^2$", 
}

# Colors for the plot lines
COLORBLIND_FRIENDLY_PALETTE = {
    'zeroshot': "#2A9D8F",          # Teal/Green
    'zeroshot-cot': "#E76F51",      # Orange/Red
    'zeroshot-2-artifacts': "#7F4CA5" # Purple
}

# --- Font Sizes and Plotting Parameters ---
TITLE_FONTSIZE = 19 # For subplot titles
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 17
GRID_LINEWIDTH = 1
PLOT_DPI = 300
PLOT_LINE_LINEWIDTH = 2.5 
PLOT_MARKER_SIZE = 8 

# --- Figure Size ---
# Directly set the desired figure size
SELECTED_FIG_SIZE = (16, 5) # <<< --- CUSTOM FIGURE SIZE ---

# --- Helper Functions for F1 Score Calculation ---
def _extract_answers_for_f1(rationales_data_sample):
    """Extracts predicted and ground truth answers from rationale data."""
    if not rationales_data_sample:
        return [], []
    pred_answers, ground_answers = [], []
    for item in rationales_data_sample:
        pred, ground = item.get('pred_answer'), item.get('ground_answer')
        if pred is not None and ground is not None and isinstance(pred, str) and isinstance(ground, str):
            pred_answers.append(pred.lower())
            ground_answers.append(ground.lower())
    return pred_answers, ground_answers

def calculate_macro_f1_score_raw(pred_answers, ground_answers):
    """Calculates macro F1 score from lists of predicted and ground truth answers."""
    if not pred_answers or not ground_answers or len(pred_answers) != len(ground_answers):
        return None
    possible_labels = ['real', 'ai-generated']
    try:
        score = f1_score(ground_answers, pred_answers, labels=possible_labels, average='macro', zero_division=0)
        return round(score * 100, 1) 
    except Exception as e:
        print(f"    Error calculating F1 score: {e}", file=sys.stderr)
        return None

def calculate_f1_from_rationales_list(rationales_list):
    """Calculates F1 score from a list of rationale objects."""
    if not rationales_list:
        return None, 0
    pred_answers, ground_answers = _extract_answers_for_f1(rationales_list)
    n_samples = len(ground_answers)
    if n_samples == 0:
        return None, 0
    f1 = calculate_macro_f1_score_raw(pred_answers, ground_answers)
    return f1, n_samples

# --- Data Loading for a Single Point (with Caching) ---
def get_f1_score(model_name, dataset_name, method_key, n_val, wait_val):
    """
    Loads or calculates F1 score for a specific set of parameters.
    Handles caching to avoid redundant calculations.
    """
    os.makedirs(F1_PLOT_CACHE_DIR, exist_ok=True)
    cache_fname_part = f"F1_{model_name}_{dataset_name}_{method_key}_n{str(n_val)}_w{wait_val}.json"
    fpath_cache = os.path.join(F1_PLOT_CACHE_DIR, cache_fname_part)

    if os.path.exists(fpath_cache):
        try:
            with open(fpath_cache, 'r') as cf:
                cached_data = json.load(cf)
            score = cached_data.get('score')
            if score is not None:
                print(f"  CACHE HIT: {model_name}/{dataset_name}/{method_key}/n={n_val}. Score: {score}")
                return score
        except Exception as e:
            print(f"  CACHE ERROR reading {fpath_cache}: {e}. Recalculating.", file=sys.stderr)

    prefix = "AI_util" if "llama" in model_name.lower() else "AI_dev"
    fname_rationale = f"{prefix}-{dataset_name}-{model_name}-{method_key}-n{str(n_val)}-wait{wait_val}-rationales.jsonl"
    fpath_rationale = os.path.join(RESPONSES_DIR, fname_rationale)
    score = None
    n_samples = 0

    if os.path.exists(fpath_rationale):
        try:
            rationales = []
            try:
                with open(fpath_rationale, 'r') as f:
                    content = json.load(f)
                if isinstance(content, dict) and 'rationales' in content and isinstance(content['rationales'], list):
                    rationales = content['rationales']
                elif isinstance(content, list):
                    rationales = content
                else:
                    raise json.JSONDecodeError("Content is not a list or expected dict structure", "", 0)
            except json.JSONDecodeError: 
                with open(fpath_rationale, 'r') as f_jsonl:
                    rationales = [json.loads(line) for line in f_jsonl if line.strip()]
            
            if not isinstance(rationales, list):
                print(f"    ERROR: Rationale file {fname_rationale} did not yield a list.", file=sys.stderr)
                rationales = []

            score, n_samples = calculate_f1_from_rationales_list(rationales)

            if score is not None:
                print(f"  CALCULATED: {model_name}/{dataset_name}/{method_key}/n={n_val}. Score: {score} (N={n_samples})")
                with open(fpath_cache, 'w') as cf:
                    json.dump({'score': score, 'n_samples': n_samples}, cf)
            else:
                print(f"    WARN: Could not calculate F1 for {fname_rationale}. N_samples={n_samples}", file=sys.stderr)
        except Exception as e:
            print(f"    ERROR processing rationale file {fname_rationale}: {e}", file=sys.stderr)
    else:
        print(f"    INFO: Rationale file NOT FOUND: {fpath_rationale}")
    return score 

# --- Main Script Logic ---
def main():
    if not os.path.isdir(RESPONSES_DIR):
        print(f"ERROR: RESPONSES_DIR '{RESPONSES_DIR}' not found. Please check the path.", file=sys.stderr)
        sys.exit(1)
    
    if TARGET_LLAMA_MODEL_NAME == "llama3-11b" and "PLEASE UPDATE THIS" in "llama3-11b": 
        print(f"WARNING: TARGET_LLAMA_MODEL_NAME is set to '{TARGET_LLAMA_MODEL_NAME}'.")
        print("Please ensure this matches the Llama 3 model name string used in your rationale filenames.")
        user_input = input(f"Continue with '{TARGET_LLAMA_MODEL_NAME}'? (yes/no): ")
        if user_input.lower() != 'yes':
            print("Exiting. Please update TARGET_LLAMA_MODEL_NAME in the script.")
            sys.exit(1)

    plot_data = []
    print(f"\n--- Collecting F1 Scores for {TARGET_LLAMA_MODEL_NAME} ---")
    for n_val in N_VALUES:
        for dataset_id, dataset_display_name in DATASETS_TO_PLOT.items():
            for method_key in METHODS_TO_PLOT: 
                print(f"Processing: n={n_val}, Dataset='{dataset_display_name}', Method='{method_key}'")
                f1 = get_f1_score(TARGET_LLAMA_MODEL_NAME, dataset_id, method_key, n_val, WAIT_VAL_PLOT)
                plot_data.append({
                    'n_value': n_val,
                    'dataset_id': dataset_id,
                    'dataset_display_name': dataset_display_name,
                    'method': method_key, 
                    'f1_score': f1 if f1 is not None else np.nan 
                })

    df_plot = pd.DataFrame(plot_data)

    if df_plot.empty or df_plot['f1_score'].isnull().all():
        print("\nNo data collected or all F1 scores are missing. Cannot generate plot.")
        sys.exit(1)

    # --- Plotting ---
    print("\n--- Generating Plot ---")
    num_datasets = len(DATASETS_TO_PLOT)
    # Use SELECTED_FIG_SIZE for figsize
    fig, axes = plt.subplots(1, num_datasets, figsize=SELECTED_FIG_SIZE) 
    if num_datasets == 1: 
        axes = [axes]

    handles_dict = {} 
    labels_for_legend_dict = {} 
    
    x_axis_title = "Sampled Responses" 

    for i, (dataset_id, dataset_display_name) in enumerate(DATASETS_TO_PLOT.items()):
        ax = axes[i]
        dataset_df = df_plot[df_plot['dataset_id'] == dataset_id]

        y_axis_min_local = None
        y_axis_max_local = None

        if not dataset_df.empty and not dataset_df['f1_score'].isnull().all():
            min_f1_local = dataset_df['f1_score'].dropna().min()
            max_f1_local = dataset_df['f1_score'].dropna().max()
            
            if pd.notna(min_f1_local) and pd.notna(max_f1_local):
                padding = (max_f1_local - min_f1_local) * 0.10 
                if padding < 2: 
                    padding = 2.0 
                y_axis_min_local = max(0, min_f1_local - padding) 
                y_axis_max_local = min(100, max_f1_local + padding) 
                if (y_axis_max_local - y_axis_min_local) < 5.0:
                    center_point = min_f1_local if min_f1_local == max_f1_local else (min_f1_local + max_f1_local) / 2.0
                    y_axis_min_local = max(0, center_point - 2.5)
                    y_axis_max_local = min(100, center_point + 2.5)
                    if y_axis_max_local - y_axis_min_local < 5.0: 
                        y_axis_max_local = min(100, y_axis_min_local + 5.0)
            else: 
                single_valid_score = dataset_df['f1_score'].dropna()
                if not single_valid_score.empty:
                    score_val = single_valid_score.iloc[0]
                    y_axis_min_local = max(0, score_val - 2.5)
                    y_axis_max_local = min(100, score_val + 2.5)
                    if y_axis_max_local - y_axis_min_local < 5.0:
                         y_axis_max_local = min(100, y_axis_min_local + 5.0)
                else: 
                    y_axis_min_local = 0
                    y_axis_max_local = 100 
        else: 
            print(f"No valid data to plot for dataset: {dataset_display_name}")
            ax.set_title(f"{dataset_display_name}\n(No data)", fontsize=TITLE_FONTSIZE)
            ax.set_xlabel(x_axis_title, fontsize=AXIS_LABEL_FONTSIZE) 
            if i == 0:
                 ax.set_ylabel("Macro F1 Score (%)", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_xticks(N_VALUES)
            ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True)) 
            ax.set_xlim(0, max(N_VALUES) + 1) 
            ax.set_ylim(0, 100) 
            continue

        for method_key in METHODS_TO_PLOT: 
            method_df = dataset_df[dataset_df['method'] == method_key].sort_values(by='n_value')
            n_value_df = pd.DataFrame({'n_value': N_VALUES})
            method_df_plotting = pd.merge(n_value_df, method_df, on='n_value', how='left')
            
            legend_label = METHOD_DISPLAY_NAME_MAP.get(method_key, method_key)

            line, = ax.plot( 
                method_df_plotting['n_value'],
                method_df_plotting['f1_score'],
                color=COLORBLIND_FRIENDLY_PALETTE.get(method_key, '#000000'), 
                marker='o', 
                linestyle='-',
                linewidth=PLOT_LINE_LINEWIDTH,
                markersize=PLOT_MARKER_SIZE 
            )
            if method_key not in handles_dict: 
                 handles_dict[method_key] = line
                 labels_for_legend_dict[method_key] = legend_label 


        ax.set_title(dataset_display_name, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel(x_axis_title, fontsize=AXIS_LABEL_FONTSIZE) 
        if i == 0: 
            ax.set_ylabel("Macro F1 Score (%)", fontsize=AXIS_LABEL_FONTSIZE) 
        
        ax.set_xticks(N_VALUES)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True)) 

        ax.set_xlim(0, max(N_VALUES) + 1) 
        
        if y_axis_min_local is not None and y_axis_max_local is not None:
            ax.set_ylim(y_axis_min_local, y_axis_max_local)
        else: 
            ax.set_ylim(0, 100)

        ax.grid(True, linestyle='--', alpha=0.7, linewidth=GRID_LINEWIDTH)

    ordered_handles = [handles_dict[m] for m in METHODS_TO_PLOT if m in handles_dict]
    ordered_labels = [labels_for_legend_dict[m] for m in METHODS_TO_PLOT if m in labels_for_legend_dict] 

    if ordered_handles: 
        fig.legend(ordered_handles, ordered_labels, loc='upper center', 
                   bbox_to_anchor=(0.5, 1.01), 
                   ncol=len(METHODS_TO_PLOT), fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout(rect=[0, 0.0, 1, 0.92]) 

    output_filename = f"self_consistency_scaling.png" 
    try:
        plt.savefig(output_filename, dpi=PLOT_DPI) 
        print(f"\nPlot saved to: {output_filename}")
    except Exception as e:
        print(f"\nError saving plot: {e}", file=sys.stderr)
    
    plt.show()

if __name__ == "__main__":
    main()
