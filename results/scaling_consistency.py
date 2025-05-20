import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
import config
from utils import helpers # Import the helpers module

# --- Configuration ---
# Use paths from config.py
SCORES_DIR = config.SCORES_DIR # Changed from RESPONSES_DIR to SCORES_DIR
F1_PLOT_CACHE_DIR = config.F1_PLOT_CACHE_DIR
TARGET_LLAMA_MODEL_NAME = "llama3-11b" # Make sure this matches filenames

N_VALUES = [1, 5, 10, 20]
# WAIT_VAL_PLOT is no longer needed for filename construction if scores.csv doesn't include it
# WAIT_VAL_PLOT = "0"

DATASETS_TO_PLOT = {
    "d32k": "D3 (2k)",
    "df402k": "DF40 (2k)",
    "genimage2k": "GenImage (2k)"
}
METHODS_TO_PLOT = ["zeroshot", "zeroshot-cot", "zeroshot-2-artifacts"]
METHOD_DISPLAY_NAME_MAP = {
    "zeroshot": "zero-shot",
    "zeroshot-cot": "zero-shot-cot",
    "zeroshot-2-artifacts": r"zero-shot-s$^2$",
}
COLORBLIND_FRIENDLY_PALETTE = {
    'zeroshot': "#2A9D8F",
    'zeroshot-cot': "#E76F51",
    'zeroshot-2-artifacts': "#7F4CA5"
}

# Font Sizes and Plotting Parameters (can be moved to a plotting_config if desired)
TITLE_FONTSIZE = 19
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 17
GRID_LINEWIDTH = 1
PLOT_DPI = 300
PLOT_LINE_LINEWIDTH = 2.5
PLOT_MARKER_SIZE = 8
SELECTED_FIG_SIZE = (16, 5)

# --- Data Loading for a Single Point (with Caching) ---
def get_f1_score(model_name, dataset_name, method_key, n_val):
    """
    Loads F1 score from a scores.csv file for a specific set of parameters.
    Handles caching to avoid redundant disk reads.
    """
    F1_PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True) # Ensure cache dir exists
    # Cache filename remains the same as it uniquely identifies the data point
    cache_fname_part = f"F1_{model_name}_{dataset_name}_{method_key}_n{str(n_val)}.json" # Removed wait_val from cache key
    fpath_cache = F1_PLOT_CACHE_DIR / cache_fname_part

    if fpath_cache.exists():
        try:
            with open(fpath_cache, 'r') as cf:
                cached_data = json.load(cf)
            score_val = cached_data.get('score')
            if score_val is not None:
                print(f"  CACHE HIT: {model_name}/{dataset_name}/{method_key}/n={n_val}. Score: {score_val}")
                return score_val
        except Exception as e:
            print(f"  CACHE ERROR reading {fpath_cache}: {e}. Recalculating.", file=sys.stderr)

    # Construct filename for scores.csv based on helpers.save_evaluation_outputs
    # Filename format: f"{model_prefix}-{dataset_name}-{model_string}-{mode_type_str}-n{num_sequences_val}-scores.csv"
    prefix = "AI_llama" if "llama" in model_name.lower() else "AI_qwen" # Assuming model_name is like TARGET_LLAMA_MODEL_NAME

    # model_string in save_evaluation_outputs is the full model name (e.g., "llama3-11b")
    # method_key here is mode_type_str in save_evaluation_outputs
    # n_val here is num_sequences_val in save_evaluation_outputs
    fname_csv = f"{prefix}-{dataset_name}-{model_name}-{method_key}-n{str(n_val)}-scores.csv"
    fpath_csv = SCORES_DIR / fname_csv
    
    score_val = None
    # n_samples is not directly available in scores.csv, set to placeholder or remove from cache
    n_samples_placeholder = 0 

    if fpath_csv.exists():
        df_score = helpers.load_scores_csv_to_dataframe(fpath_csv) # Use helper to load CSV
        if not df_score.empty:
            try:
                # Index in CSV is like 'method_key-n<n_val>'
                # Column name is 'macro_f1'
                target_index = f"{method_key}-n{str(n_val)}"
                if target_index in df_score.index:
                    raw_f1_score = df_score.loc[target_index, 'macro_f1']
                    score_val = round(float(raw_f1_score) * 100, 1) # Convert 0-1 to 0-100 scale, round
                    print(f"  CSV READ: {model_name}/{dataset_name}/{method_key}/n={n_val}. Score: {score_val}")
                    with open(fpath_cache, 'w') as cf:
                        # n_samples is not in the CSV, so we cache a placeholder or omit it
                        json.dump({'score': score_val, 'n_samples_original_source': 'scores.csv'}, cf)
                else:
                    print(f"    WARN: Index '{target_index}' not found in {fpath_csv}. Available indices: {df_score.index.tolist()}", file=sys.stderr)
            except KeyError:
                print(f"    WARN: KeyError accessing F1 score for index '{target_index}' in {fpath_csv}.", file=sys.stderr)
            except Exception as e:
                print(f"    ERROR processing {fpath_csv}: {e}", file=sys.stderr)
        else:
            # load_scores_csv_to_dataframe already logs if file not found but empty df means issue.
            if fpath_csv.exists(): # If it exists but helper returned empty
                 print(f"    WARN: scores.csv file {fpath_csv} exists but was empty or unreadable by helper.", file=sys.stderr)
            # else: File not found, already logged by helper.
    else:
        print(f"    INFO: scores.csv file NOT FOUND: {fpath_csv}")
        
    return score_val

# --- Main Script Logic ---
def main():
    if not SCORES_DIR.is_dir(): # Check SCORES_DIR
        print(f"ERROR: SCORES_DIR '{SCORES_DIR}' not found. Please check the path in config.py.", file=sys.stderr)
        sys.exit(1)

    plot_data = []
    print(f"\n--- Collecting F1 Scores for {TARGET_LLAMA_MODEL_NAME} (from scores.csv) ---")
    for n_val_loop in N_VALUES: # Renamed n_val to n_val_loop to avoid conflict
        for dataset_id, dataset_display_name in DATASETS_TO_PLOT.items():
            for method_key_loop in METHODS_TO_PLOT: # Renamed method_key
                print(f"Processing: n={n_val_loop}, Dataset='{dataset_display_name}', Method='{method_key_loop}'")
                # Pass n_val_loop to get_f1_score
                f1 = get_f1_score(TARGET_LLAMA_MODEL_NAME, dataset_id, method_key_loop, n_val_loop)
                plot_data.append({
                    'n_value': n_val_loop,
                    'dataset_id': dataset_id,
                    'dataset_display_name': dataset_display_name,
                    'method': method_key_loop,
                    'f1_score': f1 if f1 is not None else np.nan
                })

    df_plot = pd.DataFrame(plot_data)

    if df_plot.empty or df_plot['f1_score'].isnull().all():
        print("\nNo data collected or all F1 scores are missing. Cannot generate plot.")
        print("Please ensure that evaluation scripts have been run for all N_VALUES and that scores.csv files exist in the configured SCORES_DIR.")
        sys.exit(1)

    # --- Plotting --- (The plotting logic itself remains largely unchanged)
    print("\n--- Generating Plot ---")
    num_datasets = len(DATASETS_TO_PLOT)
    fig, axes = plt.subplots(1, num_datasets, figsize=SELECTED_FIG_SIZE, squeeze=False)

    handles_dict = {}
    labels_for_legend_dict = {}
    x_axis_title = "Sampled Responses (n)" # Updated x-axis title

    for i, (dataset_id, dataset_display_name) in enumerate(DATASETS_TO_PLOT.items()):
        ax = axes[0, i] 
        dataset_df = df_plot[df_plot['dataset_id'] == dataset_id]

        y_axis_min_local = None
        y_axis_max_local = None

        if not dataset_df.empty and not dataset_df['f1_score'].isnull().all():
            min_f1_local = dataset_df['f1_score'].dropna().min()
            max_f1_local = dataset_df['f1_score'].dropna().max()
            
            if pd.notna(min_f1_local) and pd.notna(max_f1_local):
                padding = max(2.0, (max_f1_local - min_f1_local) * 0.10)
                y_axis_min_local = max(0, min_f1_local - padding) 
                y_axis_max_local = min(100, max_f1_local + padding) 
                if (y_axis_max_local - y_axis_min_local) < 5.0: 
                    center_point = (min_f1_local + max_f1_local) / 2.0
                    y_axis_min_local = max(0, center_point - 2.5)
                    y_axis_max_local = min(100, center_point + 2.5)
                    if (y_axis_max_local - y_axis_min_local) < 5.0:
                         y_axis_max_local = min(100, y_axis_min_local + 5.0)
            else: 
                y_axis_min_local, y_axis_max_local = 0, 100
        else:
            print(f"No valid data to plot for dataset: {dataset_display_name}")
            ax.set_title(f"{dataset_display_name}\n(No data)", fontsize=TITLE_FONTSIZE)
            y_axis_min_local, y_axis_max_local = 0, 100


        for method_key_plot_loop in METHODS_TO_PLOT: # Renamed method_key
            method_df = dataset_df[dataset_df['method'] == method_key_plot_loop].sort_values(by='n_value')
            n_value_df = pd.DataFrame({'n_value': N_VALUES})
            method_df_plotting = pd.merge(n_value_df, method_df, on='n_value', how='left')
            
            legend_label = METHOD_DISPLAY_NAME_MAP.get(method_key_plot_loop, method_key_plot_loop)
            line, = ax.plot(
                method_df_plotting['n_value'],
                method_df_plotting['f1_score'],
                color=COLORBLIND_FRIENDLY_PALETTE.get(method_key_plot_loop, '#000000'),
                marker='o', linestyle='-',
                linewidth=PLOT_LINE_LINEWIDTH, markersize=PLOT_MARKER_SIZE
            )
            if method_key_plot_loop not in handles_dict:
                 handles_dict[method_key_plot_loop] = line
                 labels_for_legend_dict[method_key_plot_loop] = legend_label

        ax.set_title(dataset_display_name, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel(x_axis_title, fontsize=AXIS_LABEL_FONTSIZE)
        if i == 0:
            ax.set_ylabel("Macro F1 Score (%)", fontsize=AXIS_LABEL_FONTSIZE)
        
        ax.set_xticks(N_VALUES)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True)) 
        ax.set_xlim(0, max(N_VALUES) + 1)
        ax.set_ylim(y_axis_min_local, y_axis_max_local if y_axis_max_local > y_axis_min_local else y_axis_min_local + 5) 
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=GRID_LINEWIDTH)

    ordered_handles = [handles_dict[m] for m in METHODS_TO_PLOT if m in handles_dict]
    ordered_labels = [labels_for_legend_dict[m] for m in METHODS_TO_PLOT if m in labels_for_legend_dict]

    if ordered_handles:
        fig.legend(ordered_handles, ordered_labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.01), 
                   ncol=len(METHODS_TO_PLOT), fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) 

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = "self_consistency_scaling.png" # New filename
    output_filepath = config.PLOTS_DIR / output_filename

    try:
        plt.savefig(output_filepath, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"\nPlot saved to: {output_filepath}")
    except Exception as e:
        print(f"\nError saving plot: {e}", file=sys.stderr)
    plt.close(fig)

if __name__ == "__main__":
    main()