"""
Self-Consistency Scaling Analysis Script

This script generates plots showing how different prompting methods scale with 
the number of sampled responses (self-consistency). It analyzes the relationship
between performance (Macro F1-score) and the number of response samples for
zero-shot, chain-of-thought, and zero-shot-s² methods.

The script demonstrates that zero-shot-s² often scales better with multiple
samples compared to traditional prompting methods, supporting the paper's
findings about the effectiveness of task-aligned prompting.

Key Features:
- Loads evaluation scores from CSV files across multiple datasets
- Plots performance curves for different numbers of sampled responses (n=1,5,10,20)
- Compares three prompting methods: zero-shot, zero-shot-cot, zero-shot-s²
- Generates publication-ready plots with proper formatting and legends
- Uses colorblind-friendly palette for accessibility

Output:
    Saves a multi-panel plot showing scaling behavior across datasets.
    Each panel represents one dataset with method comparison curves.

Usage:
    python results/scaling_consistency.py
    
Configuration:
    Modify TARGET_LLAMA_MODEL_NAME to analyze different models.
    Adjust N_VALUES to change the self-consistency sample sizes analyzed.
    Update DATASETS_TO_PLOT to include/exclude specific datasets.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from pathlib import Path
import logging

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import config  # Configuration module
from utils import helpers  # Utility functions

# --- Logger Setup ---
helpers.setup_global_logger(config.RESULTS_SCALING_CONSISTENCY_LOG_FILE)
# Get a logger instance for this specific module
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Data source and target model
SCORES_DIR = config.SCORES_DIR
TARGET_LLAMA_MODEL_NAME = "llama3-11b"  # Model to analyze

# Self-consistency sample sizes to analyze
N_VALUES = [1, 5, 10, 20]

# Datasets to include in the analysis
DATASETS_TO_PLOT = {
    "d32k": "D3 (2k)",
    "df402k": "DF40 (2k)", 
    "genimage2k": "GenImage (2k)"
}

# Prompting methods to compare
METHODS_TO_PLOT = ["zeroshot", "zeroshot-cot", "zeroshot-2-artifacts"]

# Display names for methods (supports LaTeX formatting)
METHOD_DISPLAY_NAME_MAP = {
    "zeroshot": "zero-shot",
    "zeroshot-cot": "zero-shot-cot",
    "zeroshot-2-artifacts": r"zero-shot-s$^2$",  # LaTeX superscript for s²
}

# Colorblind-friendly color palette
COLORBLIND_FRIENDLY_PALETTE = {
    'zeroshot': "#2A9D8F",          # Teal
    'zeroshot-cot': "#E76F51",      # Coral
    'zeroshot-2-artifacts': "#7F4CA5"  # Purple
}

# =============================================================================
# PLOTTING PARAMETERS
# =============================================================================

# Font sizes for different plot elements
TITLE_FONTSIZE = 19
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 17

# Plot styling parameters
GRID_LINEWIDTH = 1
PLOT_DPI = 300
PLOT_LINE_LINEWIDTH = 2.5
PLOT_MARKER_SIZE = 8
SELECTED_FIG_SIZE = (16, 5)  # Width x Height for multi-panel plot


def get_f1_score(model_name, dataset_name, method_key, n_val):
    """
    Load F1 score from a CSV file for specific evaluation parameters.
    
    Constructs the filename based on the standard naming convention used by
    evaluation scripts and loads the macro F1-score for the specified
    method and number of samples.
    
    Args:
        model_name (str): Name of the model (e.g., "llama3-11b")
        dataset_name (str): Dataset identifier (e.g., "genimage2k")
        method_key (str): Prompting method (e.g., "zeroshot-2-artifacts")
        n_val (int): Number of sampled responses for self-consistency
        
    Returns:
        float or None: F1-score as percentage (0-100), or None if not found
        
    File Naming Convention:
        {model_prefix}-{dataset_name}-{model_name}-{method_key}-n{n_val}-scores.csv
        
    Example:
        AI_llama-genimage2k-llama3-11b-zeroshot-2-artifacts-n5-scores.csv
    """
    # Determine model prefix based on model family
    prefix = "AI_llama" if "llama" in model_name.lower() else "AI_qwen"
    
    # Construct filename following standard convention
    fname_csv = f"{prefix}-{dataset_name}-{model_name}-{method_key}-n{str(n_val)}-scores.csv"
    fpath_csv = SCORES_DIR / fname_csv
    
    score_val = None

    if fpath_csv.exists():
        # Load scores using helper function
        df_score = helpers.load_scores_csv_to_dataframe(fpath_csv)
        
        if not df_score.empty:
            try:
                # Construct expected index key
                target_index = f"{method_key}-n{str(n_val)}"
                
                if target_index in df_score.index:
                    # Extract F1 score and convert to percentage
                    raw_f1_score = df_score.loc[target_index, 'macro_f1']
                    score_val = round(float(raw_f1_score) * 100, 1)
                    logger.debug(f"Loaded F1 score: {model_name}/{dataset_name}/{method_key}/n={n_val} = {score_val}%")
                else:
                    logger.warning(f"Index '{target_index}' not found in {fpath_csv}. Available indices: {df_score.index.tolist()}")
                    
            except KeyError:
                logger.warning(f"KeyError accessing F1 score for index '{target_index}' in {fpath_csv}.")
            except Exception as e:
                logger.error(f"Error processing {fpath_csv}: {e}", exc_info=True)
        else:
            logger.warning(f"Scores CSV file {fpath_csv} exists but was empty or unreadable.")
    else:
        logger.debug(f"Scores CSV file not found: {fpath_csv}")
        
    return score_val


def main():
    """
    Main execution function for self-consistency scaling analysis.
    
    Loads F1 scores for different numbers of sampled responses across multiple
    datasets and methods, then generates a multi-panel plot showing how each
    method scales with the number of samples.
    
    The plot demonstrates the effectiveness of zero-shot-s² compared to 
    traditional zero-shot and chain-of-thought prompting when using
    self-consistency (multiple response sampling).
    
    Process:
        1. Validate that scores directory exists
        2. Load F1 scores for all combinations of n_values, datasets, and methods
        3. Create DataFrame for plotting
        4. Generate multi-panel plot with one panel per dataset
        5. Save plot to configured output directory
        
    Output:
        PNG file saved to config.PLOTS_DIR with publication-ready formatting.
    """
    # Validate scores directory exists
    if not SCORES_DIR.is_dir():
        logger.error(f"SCORES_DIR '{SCORES_DIR}' not found. Please check the path in config.py.")
        sys.exit(1)

    # =============================================================================
    # DATA COLLECTION PHASE
    # =============================================================================
    
    plot_data = []
    logger.info(f"=== Collecting F1 Scores for {TARGET_LLAMA_MODEL_NAME} ===")
    logger.info(f"N-values: {N_VALUES}")
    logger.info(f"Datasets: {list(DATASETS_TO_PLOT.keys())}")
    logger.info(f"Methods: {METHODS_TO_PLOT}")
    
    # Collect scores for all parameter combinations
    for n_val_loop in N_VALUES:
        for dataset_id, dataset_display_name in DATASETS_TO_PLOT.items():
            for method_key_loop in METHODS_TO_PLOT:
                logger.debug(f"Processing: n={n_val_loop}, Dataset='{dataset_display_name}', Method='{method_key_loop}'")
                
                # Load F1 score for this combination
                f1 = get_f1_score(TARGET_LLAMA_MODEL_NAME, dataset_id, method_key_loop, n_val_loop)
                
                # Store data point for plotting
                plot_data.append({
                    'n_value': n_val_loop,
                    'dataset_id': dataset_id,
                    'dataset_display_name': dataset_display_name,
                    'method': method_key_loop,
                    'f1_score': f1 if f1 is not None else np.nan
                })

    # Convert to DataFrame for easier manipulation
    df_plot = pd.DataFrame(plot_data)

    # Validate that we have data to plot
    if df_plot.empty or df_plot['f1_score'].isnull().all():
        logger.warning("No data collected or all F1 scores are missing. Cannot generate plot.")
        logger.warning("Please ensure that evaluation scripts have been run for all N_VALUES and that scores.csv files exist in the configured SCORES_DIR.")
        logger.warning(f"Expected files should be in: {SCORES_DIR}")
        sys.exit(1)

    # Log data collection summary
    valid_scores = df_plot['f1_score'].notna().sum()
    total_expected = len(N_VALUES) * len(DATASETS_TO_PLOT) * len(METHODS_TO_PLOT)
    logger.info(f"Data collection complete: {valid_scores}/{total_expected} scores loaded successfully")

    # =============================================================================
    # PLOT GENERATION PHASE
    # =============================================================================
    
    logger.info("=== Generating Self-Consistency Scaling Plot ===")
    
    # Set up subplot structure
    num_datasets = len(DATASETS_TO_PLOT)
    fig, axes = plt.subplots(1, num_datasets, figsize=SELECTED_FIG_SIZE, squeeze=False)

    # Storage for legend elements
    handles_dict = {}
    labels_for_legend_dict = {}
    x_axis_title = "Sampled Responses (n)"

    # Generate one subplot per dataset
    for i, (dataset_id, dataset_display_name) in enumerate(DATASETS_TO_PLOT.items()):
        ax = axes[0, i]
        dataset_df = df_plot[df_plot['dataset_id'] == dataset_id]
        
        # Calculate appropriate y-axis range based on data
        y_axis_min_local, y_axis_max_local = 0, 100  # Default values

        if not dataset_df.empty and not dataset_df['f1_score'].isnull().all():
            min_f1_local_val = dataset_df['f1_score'].dropna().min()
            max_f1_local_val = dataset_df['f1_score'].dropna().max()
            
            if pd.notna(min_f1_local_val) and pd.notna(max_f1_local_val):
                # Add padding around data range
                padding = max(2.0, (max_f1_local_val - min_f1_local_val) * 0.10)
                y_axis_min_local = max(0, min_f1_local_val - padding)
                y_axis_max_local = min(100, max_f1_local_val + padding)
                
                # Ensure minimum range for readability
                if (y_axis_max_local - y_axis_min_local) < 5.0:
                    center_point = (min_f1_local_val + max_f1_local_val) / 2.0
                    y_axis_min_local = max(0, center_point - 2.5)
                    y_axis_max_local = min(100, center_point + 2.5)
                    
                    if (y_axis_max_local - y_axis_min_local) < 5.0:  # Ensure absolute minimum range
                         y_axis_max_local = min(100, y_axis_min_local + 5.0)
        else:
            logger.info(f"No valid data to plot for dataset: {dataset_display_name}")
            ax.set_title(f"{dataset_display_name}\n(No data)", fontsize=TITLE_FONTSIZE)

        # Plot lines for each method
        for method_key_plot_loop in METHODS_TO_PLOT:
            method_df = dataset_df[dataset_df['method'] == method_key_plot_loop].sort_values(by='n_value')
            
            # Ensure all N_VALUES are present for consistent x-axis
            n_value_df = pd.DataFrame({'n_value': N_VALUES})
            method_df_plotting = pd.merge(n_value_df, method_df, on='n_value', how='left')
            
            # Get display label and color
            legend_label = METHOD_DISPLAY_NAME_MAP.get(method_key_plot_loop, method_key_plot_loop)
            color = COLORBLIND_FRIENDLY_PALETTE.get(method_key_plot_loop, '#000000')
            
            # Plot line (NaN values will create gaps automatically)
            line, = ax.plot(
                method_df_plotting['n_value'],
                method_df_plotting['f1_score'],
                color=color,
                marker='o', 
                linestyle='-',
                linewidth=PLOT_LINE_LINEWIDTH, 
                markersize=PLOT_MARKER_SIZE,
                label=legend_label
            )
            
            # Store handles for unified legend across subplots
            if method_key_plot_loop not in handles_dict:
                 handles_dict[method_key_plot_loop] = line
                 labels_for_legend_dict[method_key_plot_loop] = legend_label

        # Configure subplot appearance
        ax.set_title(dataset_display_name, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel(x_axis_title, fontsize=AXIS_LABEL_FONTSIZE)
        
        # Only leftmost subplot gets y-axis label
        if i == 0:
            ax.set_ylabel("Macro F1 Score (%)", fontsize=AXIS_LABEL_FONTSIZE)
        
        # Set axis properties
        ax.set_xticks(N_VALUES)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        ax.set_xlim(0, max(N_VALUES) + 1)
        
        # Set y-axis range (ensure max > min)
        current_y_max = y_axis_max_local if y_axis_max_local > y_axis_min_local else y_axis_min_local + 5.0
        ax.set_ylim(y_axis_min_local, current_y_max)
        
        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=GRID_LINEWIDTH)

    # =============================================================================
    # LEGEND AND LAYOUT FINALIZATION
    # =============================================================================
    
    # Create unified legend above all subplots
    ordered_handles = [handles_dict[m] for m in METHODS_TO_PLOT if m in handles_dict]
    ordered_labels = [labels_for_legend_dict[m] for m in METHODS_TO_PLOT if m in labels_for_legend_dict]

    if ordered_handles:
        fig.legend(ordered_handles, ordered_labels, 
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1.01),  # Position above subplots
                   ncol=len(METHODS_TO_PLOT), 
                   fontsize=LEGEND_FONTSIZE)
    
    # Adjust layout to accommodate legend
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    # =============================================================================
    # SAVE PLOT
    # =============================================================================
    
    # Ensure output directory exists
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename and path
    output_filename = "self_consistency_scaling_from_csv_no_cache.png"
    output_filepath = config.PLOTS_DIR / output_filename

    try:
        plt.savefig(output_filepath, dpi=PLOT_DPI, bbox_inches='tight')
        logger.info(f"Self-consistency scaling plot saved successfully: {output_filepath}")
        
        # Log plot characteristics for verification
        logger.info(f"Plot characteristics:")
        logger.info(f"  - Resolution: {PLOT_DPI} DPI")
        logger.info(f"  - Dimensions: {SELECTED_FIG_SIZE[0]}x{SELECTED_FIG_SIZE[1]} inches")
        logger.info(f"  - Datasets: {num_datasets} panels")
        logger.info(f"  - Methods compared: {len(METHODS_TO_PLOT)}")
        
    except Exception as e:
        logger.error(f"Error saving plot to {output_filepath}: {e}", exc_info=True)
        sys.exit(1)
        
    finally:
        plt.close(fig)  # Clean up memory

    logger.info("=== Self-Consistency Scaling Analysis Complete ===")


if __name__ == "__main__":
    main()