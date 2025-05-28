"""
Macro F1-Score Bar Plot Generator with Bootstrap Confidence Intervals

This script generates publication-ready bar plots comparing macro F1-scores across
different AI-generated image detection models, datasets, and prompting methods.
It includes bootstrap confidence intervals for statistical rigor and supports
both Vision-Language Models (VLMs) and traditional computer vision approaches.

The script is designed for academic publications and provides statistically
robust comparisons with proper error bars, significance annotations, and
professional formatting suitable for research papers.

Features:
- Bootstrap confidence interval calculation for statistical robustness
- Multi-model comparison (Qwen, Llama, CoDE) across datasets
- Method performance comparison with statistical significance testing
- Professional publication-ready formatting with LaTeX math notation
- Colorblind-friendly visualization with consistent color schemes
- Automatic caching of bootstrap results for efficient re-analysis
- Configurable confidence levels and bootstrap iterations

Statistical Analysis:
- Bootstrap resampling (default: 1000 iterations) for CI estimation
- Confidence intervals computed at 95% level (configurable)
- Original F1-scores calculated from rationale-level predictions
- Error bars showing confidence interval bounds
- Statistical significance annotations for method comparisons

Visualization Design:
- Bar plots with grouped comparisons by model/method
- Custom positioning for clear visual separation
- Method-specific color coding for consistency across figures
- Error bars with appropriate cap sizes
- Grid lines for easier value reading
- Professional typography with configurable font sizes

Supported Models:
- Qwen2.5 7B: Vision-language model for image analysis
- Llama 3.2 11B: Vision-language model with reasoning capabilities  
- CoDE: Traditional computer vision model trained on D3 dataset

Supported Datasets:
- D3: Diverse dataset with real and AI-generated images
- DF40: DiffusionForensics dataset focusing on diffusion models
- GenImage: Multi-generator synthetic image benchmark

Bootstrap Methodology:
- Sample with replacement from original predictions
- Recalculate macro F1-score for each bootstrap sample
- Compute percentile-based confidence intervals
- Cache results to avoid redundant computation
- Handle edge cases with insufficient data

Output Formats:
- High-resolution PNG files for publications
- PDF format available (configurable)
- Separate plots for each dataset
- Combined multi-panel figures

Performance Metrics:
- Macro F1-score: Average of F1-scores for 'real' and 'ai-generated' classes
- Handles class imbalance through macro averaging
- Bootstrap CIs provide uncertainty quantification
- Percentage differences annotated for method comparisons

Usage:
    python results/macro_f1_bars.py
    
Output Files:
    - Bar plots saved to RESULTS_OUTPUT_DIR
    - Bootstrap cache files for efficient re-runs
    - Statistical analysis logs

Configuration:
    Edit constants at top of script to adjust:
    - Bootstrap parameters (iterations, confidence level)
    - Visualization settings (colors, fonts, sizes)
    - Model and dataset selection
    - Statistical significance thresholds

Dependencies:
    - matplotlib for plotting
    - numpy for statistical calculations
    - pandas for data manipulation
    - tqdm for progress tracking during bootstrap

Note:
    Bootstrap analysis can be computationally intensive for large datasets.
    Results are cached to enable quick re-plotting with different visual
    parameters without re-running the statistical analysis.
"""

import sys
from pathlib import Path
import logging # For logging
import json
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# from scipy.stats import norm # norm was imported but not used. Removed.
# from sklearn.metrics import f1_score # f1_score is now used via helpers

# --- Project Setup ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import config
from utils import helpers

# --- Logger Setup ---
helpers.setup_global_logger(config.RESULTS_MACRO_F1_BARS_LOG_FILE)
# Get a logger instance for this specific module.
logger = logging.getLogger(__name__)

# --- Seed for reproducibility (affects bootstrap sampling) ---
np.random.seed(0)

# --- Configuration (specific to this plotting script) ---
X_AXIS_ENTITIES_INTERNAL = ["qwen25-7b", "llama3-11b", "CoDE"]
ENTITY_PLOT_DISPLAY_NAME_MAP = {
    "qwen25-7b": "Qwen2.5 7B",
    "llama3-11b": "Llama3.2 11B", # Assuming Llama3.2 based on other scripts
    "CoDE": "CoDE"
}

ALLOWED_DATASETS_PLOT = ["d3", "df40", "genimage"] # Renamed to avoid conflict
DATASET_DISPLAY_MAP_PLOT = { # Renamed
    "d3": "D3", "df40": "DF40", "genimage": "GenImage"
}

LLM_METHODS_INTERNAL_PLOT = ["zeroshot", "zeroshot-cot", "zeroshot-2-artifacts"] # Renamed
BASELINE_LLM_METHOD_PLOT = "zeroshot-cot" # Renamed
METHODS_TO_ANNOTATE_DIFF_PLOT = ["zeroshot-2-artifacts"] # Renamed
LLM_METHOD_NAME_MAPPING_PLOT = { # Renamed
    'zeroshot': 'zero-shot',
    'zeroshot-cot': 'zero-shot-cot',
    'zeroshot-2-artifacts': r'zero-shot-s$^2$'
}
COORDINATED_ASSESSMENT_METHOD_IDENTIFIER_PLOT = "_COORDINATED_ASSESSMENT_METHOD_IDENTIFIER_" # Renamed

N_VAL_PLOT = "1" # Renamed
# WAIT_VAL_PLOT = "0" # Filenames from eval likely omit -wait0

# Confidence Interval Configuration
CONFIDENCE_LEVEL_PLOT = 0.95 # Renamed
N_BOOTSTRAP_ITERATIONS_PLOT = 1000 # Renamed

# Plotting Configuration (local to this script)
COLORBLIND_FRIENDLY_PALETTE_PLOT = {"zeroshot": "#2A9D8F", "zeroshot-cot": "#E76F51", "zeroshot-2-artifacts": "#7F4CA5"} # Mapped to method keys
CODE_BAR_COLOR_PLOT = 'darkgrey'
CODE_BAR_HATCH_PLOT = '//'
CAPSIZE_FOR_ERROR_BARS_PLOT = 3
CUSTOM_X_AXIS_POSITIONS_PLOT = [0, 1, 1.75] # Ensure length matches X_AXIS_ENTITIES_INTERNAL
FIG_SIZE_PLOT = (16, 5)
TITLE_FONTSIZE_PLOT = 19
AXIS_LABEL_FONTSIZE_PLOT = 18
TICK_LABEL_FONTSIZE_PLOT = 16
LEGEND_FONTSIZE_PLOT = 17
ANNOTATION_FONTSIZE_PLOT = 13
ANNOTATION_COLOR_PLOT = 'black'
# plt.rcParams['legend.fontsize'] = LEGEND_FONTSIZE_PLOT # Set when creating legend
BAR_WIDTH_PLOT = 0.25 # Adjusted for potentially 3 LLM methods + CoDE
BAR_EDGE_COLOR_PLOT = 'black'
GRID_LINEWIDTH_PLOT = 1
TICK_WIDTH_PLOT = 1.0
TICK_LENGTH_PLOT = 5


# --- F1 Calculation Helpers (Now using utils.helpers) ---
def get_original_f1_and_n_samples_local(original_rationales_data):
    """Calculates original Macro F1 and N from the full rationales data using helpers."""
    if not original_rationales_data:
        return None, 0
    pred_answers, ground_answers = helpers._extract_answers_for_f1(original_rationales_data)
    n_samples = len(ground_answers)
    if n_samples == 0:
        return None, 0
    # helpers.calculate_macro_f1_score_from_answers returns score 0-100, rounded to 1dp
    original_score_percentage = helpers.calculate_macro_f1_score_from_answers(pred_answers, ground_answers)
    return original_score_percentage, n_samples

# --- Bootstrap CI Calculation (Specific to this script) ---
def calculate_bootstrap_ci_and_error_values_local(
    original_rationales_data,
    n_bootstrap=N_BOOTSTRAP_ITERATIONS_PLOT,
    confidence_level_plot=CONFIDENCE_LEVEL_PLOT
):
    """
    Calculates Macro F1 score and its bootstrap confidence interval.
    Uses get_original_f1_and_n_samples_local which in turn uses helpers.
    Returns the original score, error below, error above, and number of samples.
    """
    original_score_percentage, n_samples = get_original_f1_and_n_samples_local(original_rationales_data)

    if original_score_percentage is None or n_samples == 0:
        logger.debug(f"Original score is None or N=0 for CI calculation. N={n_samples}")
        return None, 0, 0, n_samples

    bootstrapped_f1_scores = []
    original_rationales_data_list = list(original_rationales_data) # For np.random.choice

    for _ in range(n_bootstrap):
        if n_samples == 0: continue # Should be caught earlier
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample_data = [original_rationales_data_list[idx] for idx in bootstrap_indices]
        
        pred_answers_boot, ground_answers_boot = helpers._extract_answers_for_f1(bootstrap_sample_data)
        if not pred_answers_boot or not ground_answers_boot:
            continue

        b_score = helpers.calculate_macro_f1_score_from_answers(pred_answers_boot, ground_answers_boot)
        if b_score is not None:
            bootstrapped_f1_scores.append(b_score)

    if not bootstrapped_f1_scores:
        logger.warning(f"No valid F1 scores from {n_bootstrap} bootstrap iterations (N={n_samples}). Original score: {original_score_percentage}. Using zero error.")
        return original_score_percentage, 0, 0, n_samples

    alpha = 1 - confidence_level_plot
    ci_lower = np.percentile(bootstrapped_f1_scores, 100 * (alpha / 2.0))
    ci_upper = np.percentile(bootstrapped_f1_scores, 100 * (1 - (alpha / 2.0)))

    error_below = original_score_percentage - ci_lower
    error_above = ci_upper - original_score_percentage
    error_below = max(0, error_below)
    error_above = max(0, error_above)
    
    return original_score_percentage, error_below, error_above, n_samples

# --- Data Loading with Caching for Bootstrap CIs (Refactored) ---
def load_data_for_f1_bars_plot(
    entities_to_load_list: list,
    datasets_to_load_list: list,
    llm_methods_to_load_list: list,
    n_val_param_plot: str
    # wait_val_param_plot: str # Likely omitted from filenames
):
    logger.info(f"Loading/Calculating Scores (Bootstrap CI) from Rationales in: {config.RESPONSES_DIR}")
    config.CI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_plot_data = []
    
    stats = {"cache_hits": 0, "bootstrapped_fresh": 0, "rationales_missing": 0, "processing_errors": 0}

    for entity_name_load in entities_to_load_list:
        for dataset_name_load in datasets_to_load_list:
            # --- CoDE Handling ---
            if entity_name_load == "CoDE":
                method_key_cache = COORDINATED_ASSESSMENT_METHOD_IDENTIFIER_PLOT
                cache_filename = f"CI_CoDE_{dataset_name_load}.json" # wait_val not in CoDE filename
                fpath_cache_plot = config.CI_CACHE_DIR / cache_filename
                
                score_plot, err_b_plot, err_a_plot, n_samp_ci_plot = None, 0, 0, 0
                
                if fpath_cache_plot.exists():
                    try:
                        with open(fpath_cache_plot, 'r') as cf: cached_data = json.load(cf)
                        score_plot = cached_data['score']
                        err_b_plot = cached_data['error_below']
                        err_a_plot = cached_data['error_above']
                        n_samp_ci_plot = cached_data['n_samples']
                        logger.info(f"  CACHE HIT: CoDE on {dataset_name_load}. Score: {score_plot if score_plot is not None else 'N/A'}")
                        stats["cache_hits"] +=1
                    except Exception as e:
                        logger.warning(f"CACHE ERROR (CoDE - {dataset_name_load}): {e}. Recalculating.", exc_info=True)
                        score_plot = None # Force recalculation
                
                if score_plot is None:
                    # Filename for CoDE rationales (adjust prefix if needed, e.g. AI_qwen or AI_dev)
                    fname_rationale_code = f"AI_qwen-{dataset_name_load}-CoDE-rationales.jsonl" # Example
                    fpath_rationale_code = config.RESPONSES_DIR / fname_rationale_code
                    
                    if fpath_rationale_code.exists():
                        rationales_list_code = helpers.load_rationales_from_file(fpath_rationale_code)
                        if rationales_list_code:
                            score_plot, err_b_plot, err_a_plot, n_samp_ci_plot = calculate_bootstrap_ci_and_error_values_local(rationales_list_code)
                            if score_plot is not None:
                                logger.info(f"  BOOTSTRAPPED: CoDE on {dataset_name_load}. Score: {score_plot:.2f} (N={n_samp_ci_plot}), Err: [{err_b_plot:.2f}, {err_a_plot:.2f}]")
                                with open(fpath_cache_plot, 'w') as cf:
                                    json.dump({'score': score_plot, 'error_below': err_b_plot, 'error_above': err_a_plot, 'n_samples': n_samp_ci_plot}, cf)
                                stats["bootstrapped_fresh"] +=1
                            else:
                                logger.warning(f"Could not calculate F1/CI for CoDE on {dataset_name_load} from {fpath_rationale_code}.")
                                stats["processing_errors"] +=1
                        else:
                            logger.warning(f"No rationales loaded by helper from {fpath_rationale_code} for CoDE.")
                            stats["processing_errors"] +=1
                    else:
                        logger.info(f"CoDE rationale file NOT FOUND: {fpath_rationale_code}")
                        stats["rationales_missing"] +=1
                
                all_plot_data.append({
                    'entity': entity_name_load, 'dataset': dataset_name_load,
                    'method': method_key_cache, # Special identifier for CoDE
                    'score': score_plot, 'error_below': err_b_plot, 'error_above': err_a_plot, 'n_samples': n_samp_ci_plot
                })

            # --- LLM Handling ---
            else:
                for llm_method_load in llm_methods_to_load_list:
                    # wait_val_param_plot is omitted from cache filename if not in actual data filenames
                    cache_filename_llm = f"CI_{entity_name_load}_{dataset_name_load}_{llm_method_load}_n{n_val_param_plot}.json"
                    fpath_cache_llm = config.CI_CACHE_DIR / cache_filename_llm
                    
                    score_llm, err_b_llm, err_a_llm, n_samp_ci_llm = None, 0, 0, 0

                    if fpath_cache_llm.exists():
                        try:
                            with open(fpath_cache_llm, 'r') as cf: cached_data_llm = json.load(cf)
                            score_llm = cached_data_llm['score']
                            err_b_llm = cached_data_llm['error_below']
                            err_a_llm = cached_data_llm['error_above']
                            n_samp_ci_llm = cached_data_llm['n_samples']
                            logger.info(f"  CACHE HIT: {entity_name_load} ({llm_method_load}) on {dataset_name_load}. Score: {score_llm if score_llm is not None else 'N/A'}")
                            stats["cache_hits"] +=1
                        except Exception as e:
                            logger.warning(f"CACHE ERROR (LLM - {entity_name_load}/{llm_method_load}/{dataset_name_load}): {e}. Recalculating.", exc_info=True)
                            score_llm = None

                    if score_llm is None:
                        prefix_llm = "AI_llama" if "llama" in entity_name_load.lower() else "AI_qwen"
                        # Assuming filenames from eval scripts omit -wait0 if wait is 0
                        fname_rationale_llm = f"{prefix_llm}-{dataset_name_load}-{entity_name_load}-{llm_method_load}-n{n_val_param_plot}-rationales.jsonl"
                        fpath_rationale_llm = config.RESPONSES_DIR / fname_rationale_llm

                        if fpath_rationale_llm.exists():
                            rationales_list_llm = helpers.load_rationales_from_file(fpath_rationale_llm)
                            if rationales_list_llm:
                                score_llm, err_b_llm, err_a_llm, n_samp_ci_llm = calculate_bootstrap_ci_and_error_values_local(rationales_list_llm)
                                if score_llm is not None:
                                    logger.info(f"  BOOTSTRAPPED: {entity_name_load} ({llm_method_load}) on {dataset_name_load}. Score: {score_llm:.2f} (N={n_samp_ci_llm}), Err: [{err_b_llm:.2f}, {err_a_llm:.2f}]")
                                    with open(fpath_cache_llm, 'w') as cf:
                                        json.dump({'score': score_llm, 'error_below': err_b_llm, 'error_above': err_a_llm, 'n_samples': n_samp_ci_llm}, cf)
                                    stats["bootstrapped_fresh"] +=1
                                else:
                                    logger.warning(f"Could not calculate F1/CI for {entity_name_load} ({llm_method_load}) on {dataset_name_load} from {fpath_rationale_llm}.")
                                    stats["processing_errors"] +=1
                            else:
                                logger.warning(f"No rationales loaded by helper from {fpath_rationale_llm} for LLM.")
                                stats["processing_errors"] +=1
                        else:
                            logger.info(f"LLM rationale file NOT FOUND: {fpath_rationale_llm}")
                            stats["rationales_missing"] +=1
                    
                    all_plot_data.append({
                        'entity': entity_name_load, 'dataset': dataset_name_load, 
                        'method': llm_method_load, 'score': score_llm,
                        'error_below': err_b_llm, 'error_above': err_a_llm, 'n_samples': n_samp_ci_llm
                    })
    
    logger.info(f"\nFinished loading/calculating F1 bar scores. Cache Hits: {stats['cache_hits']}, Bootstrapped Fresh: {stats['bootstrapped_fresh']}, Rationales Missing: {stats['rationales_missing']}, Processing Errors: {stats['processing_errors']}")
    if not all_plot_data:
        logger.warning("No data was loaded for F1 bar plot. Cannot generate plot.")
        return pd.DataFrame()
    
    df_plot_data = pd.DataFrame(all_plot_data)
    df_plot_data['score'] = pd.to_numeric(df_plot_data['score'], errors='coerce').fillna(0) # Ensure numeric, fill NaN with 0
    df_plot_data['error_below'] = pd.to_numeric(df_plot_data['error_below'], errors='coerce').fillna(0)
    df_plot_data['error_above'] = pd.to_numeric(df_plot_data['error_above'], errors='coerce').fillna(0)
    df_plot_data['n_samples'] = pd.to_numeric(df_plot_data['n_samples'], errors='coerce').fillna(0)
    
    # The 'model_family' and 'model_size' columns from original script were not used in plotting.
    # If needed, they can be re-added using helpers.parse_model_name.
    # df_plot_data['model_family'] = df_plot_data['entity'].map(MODEL_FAMILY_MAP_TABLE)
    # df_plot_data['model_size'] = df_plot_data['entity'].map(MODEL_SIZE_MAP_TABLE)
    
    df_plot_data = df_plot_data.sort_values(by=['dataset', 'entity', 'method'])
    return df_plot_data

# --- Main Script Execution ---
if __name__ == "__main__":
    if not config.RESPONSES_DIR.is_dir():
        logger.error(f"RESPONSES_DIR '{config.RESPONSES_DIR}' not found. Please check config.py. Exiting.")
        sys.exit(1)

    scores_df_plot = load_data_for_f1_bars_plot(
        X_AXIS_ENTITIES_INTERNAL,
        ALLOWED_DATASETS_PLOT,
        LLM_METHODS_INTERNAL_PLOT,
        N_VAL_PLOT
        # WAIT_VAL_PLOT # Filenames likely omit -wait0
    )

    if scores_df_plot.empty:
        logger.error("Exiting: No data available to plot (DataFrame is empty after loading). Check loading warnings.")
        sys.exit(1)
    elif scores_df_plot['score'].isnull().all() or (scores_df_plot['score'] == 0).all():
        logger.warning("All loaded scores are effectively zero or NaN. Plot may show only zero-height bars.")

    # --- Plot Generation ---
    logger.info("--- Generating F1 Bar Plot ---")
    num_datasets_plot = len(ALLOWED_DATASETS_PLOT)
    if num_datasets_plot == 0:
        logger.error("No datasets defined for plotting in ALLOWED_DATASETS_PLOT. Exiting.")
        sys.exit(1)

    fig, axes = plt.subplots(1, num_datasets_plot, figsize=FIG_SIZE_PLOT, sharey=True, squeeze=False)
    plt.rcParams['legend.fontsize'] = LEGEND_FONTSIZE_PLOT


    final_legend_handles_plot = []
    final_legend_labels_plot = []
    llm_legend_items_gathered_flag = False

    for j_dataset_idx, dataset_short_name_plot in enumerate(ALLOWED_DATASETS_PLOT):
        ax_plot = axes[0, j_dataset_idx]
        dataset_subplot_df = scores_df_plot[scores_df_plot['dataset'] == dataset_short_name_plot].copy()
        dataset_display_name_plot = DATASET_DISPLAY_MAP_PLOT.get(dataset_short_name_plot, dataset_short_name_plot.upper())
        ax_plot.set_title(dataset_display_name_plot, fontsize=TITLE_FONTSIZE_PLOT)
        
        x_positions_plot = np.array(CUSTOM_X_AXIS_POSITIONS_PLOT)
        if len(x_positions_plot) != len(X_AXIS_ENTITIES_INTERNAL):
            logger.error("CUSTOM_X_AXIS_POSITIONS_PLOT length mismatch with X_AXIS_ENTITIES_INTERNAL. Using default.")
            x_positions_plot = np.arange(len(X_AXIS_ENTITIES_INTERNAL))
        
        for i_entity_idx, entity_internal_name_plot in enumerate(X_AXIS_ENTITIES_INTERNAL):
            current_x_center_plot = x_positions_plot[i_entity_idx]
            entity_data_plot = dataset_subplot_df[dataset_subplot_df['entity'] == entity_internal_name_plot]

            if entity_data_plot.empty:
                logger.info(f"No data for entity '{entity_internal_name_plot}' in dataset '{dataset_short_name_plot}'. Skipping.")
                continue
            
            if entity_internal_name_plot == "CoDE":
                code_entry_plot = entity_data_plot[entity_data_plot['method'] == COORDINATED_ASSESSMENT_METHOD_IDENTIFIER_PLOT]
                if not code_entry_plot.empty:
                    code_score_val = code_entry_plot['score'].iloc[0]
                    err_b_val = code_entry_plot['error_below'].iloc[0]
                    err_a_val = code_entry_plot['error_above'].iloc[0]
                    yerr_val = np.array([[err_b_val], [err_a_val]])
                    ax_plot.bar(current_x_center_plot, code_score_val, BAR_WIDTH_PLOT, yerr=yerr_val,
                                capsize=CAPSIZE_FOR_ERROR_BARS_PLOT, color=CODE_BAR_COLOR_PLOT,
                                edgecolor=BAR_EDGE_COLOR_PLOT, hatch=CODE_BAR_HATCH_PLOT)
            else: # LLM
                num_llm_methods_plot = len(LLM_METHODS_INTERNAL_PLOT)
                # Adjust bar width based on number of LLM methods to prevent overlap
                current_bar_width = BAR_WIDTH_PLOT / num_llm_methods_plot * 0.8 # Make them a bit thinner than slot
                
                group_total_width = current_bar_width * num_llm_methods_plot
                start_offset_plot = - (group_total_width / 2) + (current_bar_width / 2)

                baseline_score_entry = entity_data_plot[entity_data_plot['method'] == BASELINE_LLM_METHOD_PLOT]['score']
                llm_baseline_score_val = baseline_score_entry.iloc[0] if not baseline_score_entry.empty and pd.notna(baseline_score_entry.iloc[0]) else 0.0

                for k_method_idx, llm_method_key_plot in enumerate(LLM_METHODS_INTERNAL_PLOT):
                    method_data_plot = entity_data_plot[entity_data_plot['method'] == llm_method_key_plot]
                    if not method_data_plot.empty:
                        score_val = method_data_plot['score'].iloc[0]
                        err_b_val = method_data_plot['error_below'].iloc[0]
                        err_a_val = method_data_plot['error_above'].iloc[0]
                        yerr_val = np.array([[err_b_val], [err_a_val]])
                        
                        bar_x_position = current_x_center_plot + start_offset_plot + k_method_idx * current_bar_width
                        legend_label_str = LLM_METHOD_NAME_MAPPING_PLOT.get(llm_method_key_plot, llm_method_key_plot)
                        
                        bar_color = COLORBLIND_FRIENDLY_PALETTE_PLOT.get(llm_method_key_plot, "#000000") # Fallback black
                        
                        rects_plot = ax_plot.bar(bar_x_position, score_val, current_bar_width, yerr=yerr_val,
                                                 capsize=CAPSIZE_FOR_ERROR_BARS_PLOT, label=legend_label_str,
                                                 color=bar_color, edgecolor=BAR_EDGE_COLOR_PLOT)
                        
                        if not llm_legend_items_gathered_flag and legend_label_str not in final_legend_labels_plot:
                            final_legend_handles_plot.append(rects_plot[0])
                            final_legend_labels_plot.append(legend_label_str)

                        if llm_method_key_plot in METHODS_TO_ANNOTATE_DIFF_PLOT and not (score_val == 0 and llm_baseline_score_val == 0):
                            diff_text_str = ""
                            if llm_baseline_score_val != 0:
                                rel_diff_val = ((score_val - llm_baseline_score_val) / llm_baseline_score_val) * 100
                                diff_text_str = f"+{rel_diff_val:.0f}%" if rel_diff_val >= 0 else f"{rel_diff_val:.0f}%"
                            elif score_val > 0:
                                diff_text_str = "N/A" # Baseline is 0, score is positive
                            
                            if diff_text_str:
                                text_y_pos = max(score_val, 0) + err_a_val + 2
                                ax_plot.text(bar_x_position, text_y_pos, diff_text_str, ha='center', va='bottom',
                                             fontsize=ANNOTATION_FONTSIZE_PLOT, color=ANNOTATION_COLOR_PLOT)
            if final_legend_handles_plot and not llm_legend_items_gathered_flag:
                llm_legend_items_gathered_flag = True # Gather LLM legend items only once

        if j_dataset_idx == 0:
            ax_plot.set_ylabel('Macro F1 Score (%)', fontsize=AXIS_LABEL_FONTSIZE_PLOT)
        
        ax_plot.set_xticks(x_positions_plot)
        ax_plot.set_xticklabels([ENTITY_PLOT_DISPLAY_NAME_MAP.get(e, e) for e in X_AXIS_ENTITIES_INTERNAL],
                               fontsize=TICK_LABEL_FONTSIZE_PLOT, rotation=0)
        ax_plot.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE_PLOT, width=TICK_WIDTH_PLOT, length=TICK_LENGTH_PLOT)
        ax_plot.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE_PLOT, width=TICK_WIDTH_PLOT, length=TICK_LENGTH_PLOT)
        ax_plot.grid(axis='y', linestyle='--', alpha=0.7, linewidth=GRID_LINEWIDTH_PLOT)
        ax_plot.set_ylim(0, 105) # Fixed Y-axis range

    # Add CoDE to legend if it was plotted
    if "CoDE" in X_AXIS_ENTITIES_INTERNAL:
        code_patch_legend = mpatches.Patch(facecolor=CODE_BAR_COLOR_PLOT, edgecolor=BAR_EDGE_COLOR_PLOT,
                                     hatch=CODE_BAR_HATCH_PLOT, label='CoDE')
        # Avoid duplicate CoDE legend entry if already somehow added
        if not any(isinstance(handle, mpatches.Patch) and handle.get_label() == 'CoDE' for handle in final_legend_handles_plot):
             # Construct a more descriptive label for CoDE if desired
            code_legend_label = ENTITY_PLOT_DISPLAY_NAME_MAP.get("CoDE", "CoDE")
            if COORDINATED_ASSESSMENT_METHOD_IDENTIFIER_PLOT in scores_df_plot['method'].unique(): # Check if CoDE data actually exists
                code_legend_label += ' (trained on D3)' # Add detail if CoDE was processed
            
            final_legend_handles_plot.append(code_patch_legend)
            final_legend_labels_plot.append(code_legend_label)


    if final_legend_handles_plot:
        fig.legend(final_legend_handles_plot, final_legend_labels_plot, loc='upper center',
                   ncol=len(final_legend_labels_plot), bbox_to_anchor=(0.5, 1.0), # Adjusted y for visibility
                   fontsize=LEGEND_FONTSIZE_PLOT)
    
    plt.tight_layout(rect=[0, 0, 1, 0.90]) # Adjust rect to make space for legend

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_filename_bars = "ai_generation_macro_f1_bootstrap_ci_plot_refactored.png"
    output_filepath_bars = config.PLOTS_DIR / output_filename_bars

    try:
        plt.savefig(output_filepath_bars, dpi=300, bbox_inches='tight')
        logger.info(f"--- Plot Generation Complete --- Plot saved to {output_filepath_bars}")
    except Exception as e:
        logger.error(f"Error saving F1 bar plot: {e}", exc_info=True)
    plt.close(fig) # Close the figure