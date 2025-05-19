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
RESPONSES_DIR = config.RESPONSES_DIR
F1_PLOT_CACHE_DIR = config.F1_PLOT_CACHE_DIR
TARGET_LLAMA_MODEL_NAME = "llama3-11b" # Make sure this matches filenames

N_VALUES = [1, 5, 10, 20]
WAIT_VAL_PLOT = "0"

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
def get_f1_score(model_name, dataset_name, method_key, n_val, wait_val):
    """
    Loads or calculates F1 score for a specific set of parameters using helpers.
    Handles caching to avoid redundant calculations.
    """
    F1_PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True) # Ensure cache dir exists
    cache_fname_part = f"F1_{model_name}_{dataset_name}_{method_key}_n{str(n_val)}_w{wait_val}.json"
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

    prefix = "AI_llama" if "llama" in model_name.lower() else "AI_qwen"
    # Note: The original filenames in scaling_consistency included '-wait{wait_val}'.
    # The eval scripts now save without '-wait{wait_val}'. Adjust if filenames differ.
    fname_rationale = f"{prefix}-{dataset_name}-{model_name}-{method_key}-n{str(n_val)}-rationales.jsonl"
    # If your filenames from eval scripts *do* include 'wait0', then:
    # fname_rationale = f"{prefix}-{dataset_name}-{model_name}-{method_key}-n{str(n_val)}-wait{wait_val}-rationales.jsonl"
    
    fpath_rationale = RESPONSES_DIR / fname_rationale
    
    score_val = None
    n_samples = 0

    if fpath_rationale.exists():
        # Use helper to load rationales
        rationales_list = helpers.load_rationales_from_file(fpath_rationale)
        
        if rationales_list:
            # Use helper to calculate F1 from the list of rationales
            # The possible_labels defaults to ['real', 'ai-generated'] in the helper.
            score_val, n_samples = helpers.calculate_f1_from_rationales(rationales_list)

            if score_val is not None:
                # Score from helper is already 0-100 and rounded.
                print(f"  CALCULATED: {model_name}/{dataset_name}/{method_key}/n={n_val}. Score: {score_val} (N={n_samples})")
                with open(fpath_cache, 'w') as cf:
                    json.dump({'score': score_val, 'n_samples': n_samples}, cf)
            else:
                print(f"    WARN: Could not calculate F1 for {fname_rationale} using helpers. N_samples from helper: {n_samples}", file=sys.stderr)
        else:
            print(f"    WARN: No rationales loaded by helper from {fpath_rationale}.", file=sys.stderr)
    else:
        print(f"    INFO: Rationale file NOT FOUND: {fpath_rationale}")
    return score_val

# --- Main Script Logic ---
def main():
    if not RESPONSES_DIR.is_dir():
        print(f"ERROR: RESPONSES_DIR '{RESPONSES_DIR}' not found. Please check the path in config.py.", file=sys.stderr)
        sys.exit(1)

    # (TARGET_LLAMA_MODEL_NAME check can remain if desired)

    plot_data = []
    print(f"\n--- Collecting F1 Scores for {TARGET_LLAMA_MODEL_NAME} (using helpers) ---")
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

    # --- Plotting --- (The plotting logic itself remains largely unchanged)
    print("\n--- Generating Plot ---")
    num_datasets = len(DATASETS_TO_PLOT)
    fig, axes = plt.subplots(1, num_datasets, figsize=SELECTED_FIG_SIZE, squeeze=False)
    # if num_datasets == 1: # squeeze=False handles this, axes[0,0]
    #     axes = [axes] # This line is not needed with squeeze=False

    handles_dict = {}
    labels_for_legend_dict = {}
    x_axis_title = "Sampled Responses"

    for i, (dataset_id, dataset_display_name) in enumerate(DATASETS_TO_PLOT.items()):
        ax = axes[0, i] # Access subplot correctly
        dataset_df = df_plot[df_plot['dataset_id'] == dataset_id]

        y_axis_min_local = None
        y_axis_max_local = None

        # Dynamic Y-axis scaling logic (can be kept or simplified)
        if not dataset_df.empty and not dataset_df['f1_score'].isnull().all():
            min_f1_local = dataset_df['f1_score'].dropna().min()
            max_f1_local = dataset_df['f1_score'].dropna().max()
            
            if pd.notna(min_f1_local) and pd.notna(max_f1_local):
                padding = max(2.0, (max_f1_local - min_f1_local) * 0.10)
                y_axis_min_local = max(0, min_f1_local - padding) 
                y_axis_max_local = min(100, max_f1_local + padding) 
                if (y_axis_max_local - y_axis_min_local) < 5.0: # Ensure a minimum range
                    center_point = (min_f1_local + max_f1_local) / 2.0
                    y_axis_min_local = max(0, center_point - 2.5)
                    y_axis_max_local = min(100, center_point + 2.5)
                    if (y_axis_max_local - y_axis_min_local) < 5.0:
                         y_axis_max_local = min(100, y_axis_min_local + 5.0)
            # (Simplified the single point / no data fallback for brevity, original logic was more detailed)
            else: 
                y_axis_min_local, y_axis_max_local = 0, 100
        else:
            print(f"No valid data to plot for dataset: {dataset_display_name}")
            ax.set_title(f"{dataset_display_name}\n(No data)", fontsize=TITLE_FONTSIZE)
            y_axis_min_local, y_axis_max_local = 0, 100


        for method_key in METHODS_TO_PLOT:
            method_df = dataset_df[dataset_df['method'] == method_key].sort_values(by='n_value')
            # Ensure all N_VALUES are present for plotting, merging with NaNs if data is missing
            n_value_df = pd.DataFrame({'n_value': N_VALUES})
            method_df_plotting = pd.merge(n_value_df, method_df, on='n_value', how='left')
            
            legend_label = METHOD_DISPLAY_NAME_MAP.get(method_key, method_key)
            line, = ax.plot(
                method_df_plotting['n_value'],
                method_df_plotting['f1_score'],
                color=COLORBLIND_FRIENDLY_PALETTE.get(method_key, '#000000'),
                marker='o', linestyle='-',
                linewidth=PLOT_LINE_LINEWIDTH, markersize=PLOT_MARKER_SIZE
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
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True)) # Limit y-ticks
        ax.set_xlim(0, max(N_VALUES) + 1)
        ax.set_ylim(y_axis_min_local, y_axis_max_local if y_axis_max_local > y_axis_min_local else y_axis_min_local + 5) # Ensure positive range
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=GRID_LINEWIDTH)

    ordered_handles = [handles_dict[m] for m in METHODS_TO_PLOT if m in handles_dict]
    ordered_labels = [labels_for_legend_dict[m] for m in METHODS_TO_PLOT if m in labels_for_legend_dict]

    if ordered_handles:
        fig.legend(ordered_handles, ordered_labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.01), # Adjust y to ensure it's above plots
                   ncol=len(METHODS_TO_PLOT), fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect to make space for legend and x-labels

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = "self_consistency_scaling.png"
    output_filepath = config.PLOTS_DIR / output_filename

    try:
        plt.savefig(output_filepath, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"\nPlot saved to: {output_filepath}")
    except Exception as e:
        print(f"\nError saving plot: {e}", file=sys.stderr)
    plt.close(fig)

if __name__ == "__main__":
    main()