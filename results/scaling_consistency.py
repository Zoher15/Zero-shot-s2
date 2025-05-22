import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from pathlib import Path
import logging # Added for logging

project_root = Path(__file__).resolve().parent.parent
# Ensure project_root is added to sys.path only once
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import config # Your config file
from utils import helpers # Your helpers module

# --- Logger Setup ---
helpers.setup_global_logger(config.RESULTS_SCALING_CONSISTENCY_LOG_FILE)
# Get a logger instance for this specific module.
logger = logging.getLogger(__name__)

# --- Configuration ---
SCORES_DIR = config.SCORES_DIR
TARGET_LLAMA_MODEL_NAME = "llama3-11b"
N_VALUES = [1, 5, 10, 20]
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

# Font Sizes and Plotting Parameters
TITLE_FONTSIZE = 19
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 17
GRID_LINEWIDTH = 1
PLOT_DPI = 300
PLOT_LINE_LINEWIDTH = 2.5
PLOT_MARKER_SIZE = 8
SELECTED_FIG_SIZE = (16, 5)

# --- Data Loading for a Single Point ---
def get_f1_score(model_name, dataset_name, method_key, n_val):
    """
    Loads F1 score directly from a scores.csv file for a specific set of parameters.
    """
    prefix = "AI_llama" if "llama" in model_name.lower() else "AI_qwen"
    fname_csv = f"{prefix}-{dataset_name}-{model_name}-{method_key}-n{str(n_val)}-scores.csv"
    fpath_csv = SCORES_DIR / fname_csv
    
    score_val = None

    if fpath_csv.exists():
        df_score = helpers.load_scores_csv_to_dataframe(fpath_csv) #
        if not df_score.empty:
            try:
                target_index = f"{method_key}-n{str(n_val)}"
                if target_index in df_score.index:
                    raw_f1_score = df_score.loc[target_index, 'macro_f1']
                    score_val = round(float(raw_f1_score) * 100, 1)
                    logger.debug(f"CSV READ: {model_name}/{dataset_name}/{method_key}/n={n_val}. Score: {score_val}")
                else:
                    logger.warning(f"Index '{target_index}' not found in {fpath_csv}. Available indices: {df_score.index.tolist()}")
            except KeyError:
                logger.warning(f"KeyError accessing F1 score for index '{target_index}' in {fpath_csv}.")
            except Exception as e:
                logger.error(f"Error processing {fpath_csv}: {e}", exc_info=True)
        else:
            if fpath_csv.exists():
                 logger.warning(f"scores.csv file {fpath_csv} exists but was empty or unreadable by helper.")
    else:
        logger.info(f"scores.csv file NOT FOUND: {fpath_csv}")
        
    return score_val

# --- Main Script Logic ---
def main():
    if not SCORES_DIR.is_dir():
        logger.error(f"SCORES_DIR '{SCORES_DIR}' not found. Please check the path in config.py.") #
        sys.exit(1)

    plot_data = []
    logger.info(f"--- Collecting F1 Scores for {TARGET_LLAMA_MODEL_NAME} (from scores.csv, no caching) ---")
    for n_val_loop in N_VALUES:
        for dataset_id, dataset_display_name in DATASETS_TO_PLOT.items():
            for method_key_loop in METHODS_TO_PLOT:
                logger.info(f"Processing: n={n_val_loop}, Dataset='{dataset_display_name}', Method='{method_key_loop}'")
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
        logger.warning("No data collected or all F1 scores are missing. Cannot generate plot.")
        logger.warning("Please ensure that evaluation scripts have been run for all N_VALUES and that scores.csv files exist in the configured SCORES_DIR.")
        sys.exit(1)

    logger.info("--- Generating Plot ---")
    num_datasets = len(DATASETS_TO_PLOT)
    fig, axes = plt.subplots(1, num_datasets, figsize=SELECTED_FIG_SIZE, squeeze=False)

    handles_dict = {}
    labels_for_legend_dict = {}
    x_axis_title = "Sampled Responses (n)"

    for i, (dataset_id, dataset_display_name) in enumerate(DATASETS_TO_PLOT.items()):
        ax = axes[0, i]
        dataset_df = df_plot[df_plot['dataset_id'] == dataset_id]
        y_axis_min_local, y_axis_max_local = 0, 100 # Default values

        if not dataset_df.empty and not dataset_df['f1_score'].isnull().all():
            min_f1_local_val = dataset_df['f1_score'].dropna().min()
            max_f1_local_val = dataset_df['f1_score'].dropna().max()
            
            if pd.notna(min_f1_local_val) and pd.notna(max_f1_local_val):
                padding = max(2.0, (max_f1_local_val - min_f1_local_val) * 0.10)
                y_axis_min_local = max(0, min_f1_local_val - padding)
                y_axis_max_local = min(100, max_f1_local_val + padding)
                if (y_axis_max_local - y_axis_min_local) < 5.0:
                    center_point = (min_f1_local_val + max_f1_local_val) / 2.0
                    y_axis_min_local = max(0, center_point - 2.5)
                    y_axis_max_local = min(100, center_point + 2.5)
                    if (y_axis_max_local - y_axis_min_local) < 5.0: # Ensure minimum range
                         y_axis_max_local = min(100, y_axis_min_local + 5.0)
        else:
            logger.info(f"No valid data to plot for dataset: {dataset_display_name}")
            ax.set_title(f"{dataset_display_name}\n(No data)", fontsize=TITLE_FONTSIZE)
            # Keep default y_axis_min_local, y_axis_max_local = 0, 100

        for method_key_plot_loop in METHODS_TO_PLOT:
            method_df = dataset_df[dataset_df['method'] == method_key_plot_loop].sort_values(by='n_value')
            n_value_df = pd.DataFrame({'n_value': N_VALUES}) # Ensure all N_VALUES are present for plotting
            method_df_plotting = pd.merge(n_value_df, method_df, on='n_value', how='left')
            
            legend_label = METHOD_DISPLAY_NAME_MAP.get(method_key_plot_loop, method_key_plot_loop)
            line, = ax.plot(
                method_df_plotting['n_value'],
                method_df_plotting['f1_score'], # NaNs will create gaps in the line
                color=COLORBLIND_FRIENDLY_PALETTE.get(method_key_plot_loop, '#000000'),
                marker='o', linestyle='-',
                linewidth=PLOT_LINE_LINEWIDTH, markersize=PLOT_MARKER_SIZE
            )
            if method_key_plot_loop not in handles_dict: # Store handles for unified legend
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
        # Ensure y_axis_max_local is greater than y_axis_min_local for set_ylim
        current_y_max = y_axis_max_local if y_axis_max_local > y_axis_min_local else y_axis_min_local + 5.0
        ax.set_ylim(y_axis_min_local, current_y_max)
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=GRID_LINEWIDTH)

    # Create a single legend for the entire figure
    ordered_handles = [handles_dict[m] for m in METHODS_TO_PLOT if m in handles_dict]
    ordered_labels = [labels_for_legend_dict[m] for m in METHODS_TO_PLOT if m in labels_for_legend_dict]

    if ordered_handles:
        fig.legend(ordered_handles, ordered_labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.01), # Adjust y to be above subplots
                   ncol=len(METHODS_TO_PLOT), fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect to make space for legend

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_filename = "self_consistency_scaling_from_csv_no_cache.png"
    output_filepath = config.PLOTS_DIR / output_filename

    try:
        plt.savefig(output_filepath, dpi=PLOT_DPI, bbox_inches='tight')
        logger.info(f"Plot saved to: {output_filepath}")
    except Exception as e:
        logger.error(f"Error saving plot: {e}", exc_info=True)
    plt.close(fig)

if __name__ == "__main__":
    main()