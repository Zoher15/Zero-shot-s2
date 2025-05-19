import os
import pandas as pd
import re
import numpy as np
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from scipy.stats import norm
from sklearn.metrics import f1_score
from pathlib import Path # <--- ADD

# Assuming config.py is in the project root (parent of 'results')
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
import config # <--- ADD

# --- Seed for reproducibility (affects bootstrap sampling) ---
np.random.seed(0)

# --- Configuration ---
RESPONSES_DIR = config.RESPONSES_DIR # <--- CHANGED
CI_CACHE_DIR = config.CI_CACHE_DIR # <--- CHANGED

X_AXIS_ENTITIES_INTERNAL = ["qwen25-7b", "llama3-11b", "CoDE"]
ENTITY_PLOT_DISPLAY_NAME_MAP = {
    "qwen25-7b": "Qwen2.5 7B",
    "llama3-11b": "Llama3.2 11B",
    "CoDE": "CoDE"
}

ALLOWED_DATASETS = ["d3", "df40", "genimage"]
DATASET_DISPLAY_MAP = {
    "d3": "D3",
    "df40": "DF40",
    "genimage": "GenImage"
}

LLM_METHODS_INTERNAL = ["zeroshot", "zeroshot-cot", "zeroshot-2-artifacts"]
# Changed baseline to zeroshot-cot
BASELINE_LLM_METHOD = "zeroshot-cot" 
# Define which method(s) should show the annotation relative to the baseline
METHODS_TO_ANNOTATE_DIFF = ["zeroshot-2-artifacts"] 
LLM_METHOD_NAME_MAPPING = {
    'zeroshot': 'zero-shot',
    'zeroshot-cot': 'zero-shot-cot',
    'zeroshot-2-artifacts': r'zero-shot-s$^2$'
}
COORDINATED_ASSESSMENT_METHOD_IDENTIFIER = "_COORDINATED_ASSESSMENT_METHOD_IDENTIFIER_"

N_VAL = "1"
WAIT_VAL = "0"

# --- Confidence Interval Configuration ---
CONFIDENCE_LEVEL = 0.95
N_BOOTSTRAP_ITERATIONS = 1000 # Number of bootstrap samples to generate

# --- Model Parsing (Informational) ---
# (Model parsing logic remains unchanged, kept for context if needed elsewhere)
MODEL_FAMILIES_DETECTED = []
MODEL_FAMILY_MAP = {}
MODEL_SIZE_MAP = {}
MODELS_BY_FAMILY = defaultdict(list)
print("--- Model Parsing (Informational) ---")
POTENTIAL_LLMS_FOR_PARSING = ["llama3-11b", "qwen25-7b"] # Simplified
for llm_name in POTENTIAL_LLMS_FOR_PARSING:
    if llm_name == "CoDE": continue
    match = re.match(r'([a-zA-Z]+)(\d+(\.\d+)?)?-(\d+[bB])', llm_name)
    if match:
        base_name = match.group(1); version = match.group(2); size = match.group(4).lower()
        family_name = base_name
        if base_name == 'llama': family_name = 'llama3.2'
        elif version and base_name == 'qwen' and version == '25': family_name = 'qwen2.5'
        elif version: family_name = f"{base_name}{version}"
        if family_name not in MODEL_FAMILIES_DETECTED: MODEL_FAMILIES_DETECTED.append(family_name)
        MODEL_FAMILY_MAP[llm_name] = family_name; MODEL_SIZE_MAP[llm_name] = size
        MODELS_BY_FAMILY[family_name].append(llm_name)
print("--- End Model Parsing ---")

# --- Plotting Configuration ---
COLORBLIND_FRIENDLY_PALETTE = ["#2A9D8F", "#E76F51", "#7F4CA5", "#F4A261"]
CODE_BAR_COLOR = 'darkgrey'
CODE_BAR_HATCH = '//'
CAPSIZE_FOR_ERROR_BARS = 3
CUSTOM_X_AXIS_POSITIONS = [0, 1, 1.75]
FIG_SIZE = (16, 5)
TITLE_FONTSIZE = 19
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 17
ANNOTATION_FONTSIZE = 13
ANNOTATION_COLOR = 'black'
plt.rcParams['legend.fontsize'] = LEGEND_FONTSIZE
BAR_WIDTH = 0.3
BAR_EDGE_COLOR = 'black'
GRID_LINEWIDTH = 1
TICK_WIDTH = 1.0
TICK_LENGTH = 5

# --- Helper Functions for F1 Calculation ---
def _extract_answers_for_f1(rationales_data_sample):
    """Extracts prediction and ground truth answers from a sample of rationales data."""
    if not rationales_data_sample:
        return [], []
    pred_answers = []
    ground_answers = []
    for item in rationales_data_sample:
        pred = item.get('pred_answer')
        ground = item.get('ground_answer')
        if pred is not None and ground is not None and isinstance(pred, str) and isinstance(ground, str):
            pred_answers.append(pred.lower())
            ground_answers.append(ground.lower())
    return pred_answers, ground_answers

def calculate_macro_f1_score(pred_answers, ground_answers):
    """Calculates Macro F1 score given lists of predictions and ground truths."""
    if not pred_answers or not ground_answers or len(pred_answers) != len(ground_answers):
        return None
    
    possible_labels = ['real', 'ai-generated']
    try:
        score = f1_score(ground_answers, pred_answers, labels=possible_labels, average='macro', zero_division=0)
        return round(score * 100, 2)
    except Exception as e:
        # print(f"Debug: Error in f1_score calculation: {e} for {len(ground_answers)} ground samples.")
        return None

def get_original_f1_and_n_samples(original_rationales_data):
    """Calculates original Macro F1 and N from the full rationales data."""
    if not original_rationales_data:
        return None, 0
    pred_answers, ground_answers = _extract_answers_for_f1(original_rationales_data)
    n_samples = len(ground_answers)
    if n_samples == 0:
        return None, 0
    original_score = calculate_macro_f1_score(pred_answers, ground_answers)
    return original_score, n_samples

# --- Bootstrap CI Calculation ---
def calculate_bootstrap_ci_and_error_values(original_rationales_data, 
                                            n_bootstrap=N_BOOTSTRAP_ITERATIONS, 
                                            confidence_level=CONFIDENCE_LEVEL):
    """
    Calculates Macro F1 score and its bootstrap confidence interval.
    Returns the original score, error below, error above, and number of samples.
    """
    original_score_percentage, n_samples = get_original_f1_and_n_samples(original_rationales_data)

    if original_score_percentage is None or n_samples == 0:
        # print(f"Debug: Original score is None or N=0. N={n_samples}")
        return None, 0, 0, n_samples 

    bootstrapped_f1_scores = []
    # Ensure original_rationales_data is a list for np.random.choice on its elements
    original_rationales_data_list = list(original_rationales_data)

    for i in range(n_bootstrap):
        # Create bootstrap sample by choosing indices with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        # Create the actual data sample based on these indices
        bootstrap_sample_data = [original_rationales_data_list[idx] for idx in bootstrap_indices]
        
        # Extract answers for this bootstrap sample
        pred_answers_boot, ground_answers_boot = _extract_answers_for_f1(bootstrap_sample_data)
        
        if not pred_answers_boot or not ground_answers_boot: # Skip if bootstrap sample is problematic
            # print(f"Debug: Bootstrap sample {i} yielded no valid answers.")
            continue

        b_score = calculate_macro_f1_score(pred_answers_boot, ground_answers_boot)
        if b_score is not None:
            bootstrapped_f1_scores.append(b_score)
        # else:
            # print(f"Debug: Bootstrap sample {i} F1 score is None.")


    if not bootstrapped_f1_scores:
        print(f"  Warning: No valid F1 scores from {n_bootstrap} bootstrap iterations (N={n_samples}). Original score: {original_score_percentage:.2f}. Using zero error.")
        return original_score_percentage, 0, 0, n_samples

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrapped_f1_scores, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrapped_f1_scores, 100 * (1 - alpha / 2))

    error_below = original_score_percentage - ci_lower
    error_above = ci_upper - original_score_percentage

    error_below = max(0, error_below) # Ensure non-negative
    error_above = max(0, error_above) # Ensure non-negative
    
    # print(f"Debug: Orig: {original_score_percentage}, CI: [{ci_lower:.2f}, {ci_upper:.2f}], ErrB: {error_below:.2f}, ErrA: {error_above:.2f}")
    return original_score_percentage, error_below, error_above, n_samples

# --- Data Loading with Caching for Bootstrap CIs ---
def load_data_for_plotting(responses_dir, entities_to_load, datasets_to_load, llm_methods_to_load, n_val_param, wait_val_param):
    print(f"\n--- Loading/Calculating Scores (Bootstrap CI) from Rationales in: {responses_dir} ---")
    os.makedirs(CI_CACHE_DIR, exist_ok=True) # Ensure cache directory exists
    all_data = []
    
    # Counters for reporting
    files_processed_bootstrap = 0
    files_loaded_from_cache = 0
    files_missing_rationales = 0
    files_error_processing = 0


    for entity_name in entities_to_load:
        for dataset_name in datasets_to_load:
            # --- CoDE Handling ---
            if entity_name == "CoDE":
                method_key_for_cache = COORDINATED_ASSESSMENT_METHOD_IDENTIFIER # Consistent key for CoDE
                cache_filename = f"CI_CoDE_{dataset_name}.json"
                fpath_cache = os.path.join(CI_CACHE_DIR, cache_filename)
                
                score, error_below, error_above, n_samples_ci = None, 0, 0, 0
                
                if os.path.exists(fpath_cache):
                    try:
                        with open(fpath_cache, 'r') as cf:
                            cached_data = json.load(cf)
                        score = cached_data['score']
                        error_below = cached_data['error_below']
                        error_above = cached_data['error_above']
                        n_samples_ci = cached_data['n_samples']
                        print(f"  CACHE HIT: CoDE on {dataset_name}. Score: {score if score is not None else 'N/A'}")
                        files_loaded_from_cache +=1
                    except Exception as e:
                        print(f"  CACHE ERROR (CoDE - {dataset_name}): {e}. Recalculating.")
                        score = None # Force recalculation
                
                if score is None: # Not in cache or cache error
                    fname_rationale = f"AI_qwen-{dataset_name}-CoDE-rationales.jsonl"
                    fpath_rationale = os.path.join(responses_dir, fname_rationale)
                    if os.path.exists(fpath_rationale):
                        try:
                            with open(fpath_rationale, 'r') as f:
                                rationales = json.load(f)
                            if not isinstance(rationales, list):
                                print(f"    ERROR: CoDE rationale file {fname_rationale} not a JSON list.")
                                rationales = [] ; files_error_processing +=1
                            
                            score, error_below, error_above, n_samples_ci = calculate_bootstrap_ci_and_error_values(rationales)
                            if score is not None:
                                print(f"  BOOTSTRAPPED: CoDE on {dataset_name}. Score: {score:.2f} (N={n_samples_ci}), Err: [{error_below:.2f}, {error_above:.2f}]")
                                with open(fpath_cache, 'w') as cf:
                                    json.dump({'score': score, 'error_below': error_below, 'error_above': error_above, 'n_samples': n_samples_ci}, cf)
                                files_processed_bootstrap +=1
                            else:
                                print(f"    WARN: Could not calculate F1/CI for CoDE on {dataset_name} from {fname_rationale}.")
                                files_error_processing +=1
                        except Exception as e:
                            print(f"    ERROR processing CoDE file {fname_rationale}: {e}")
                            files_error_processing +=1
                    else:
                        print(f"    Info: CoDE rationale file NOT FOUND: {fpath_rationale}")
                        files_missing_rationales +=1
                
                all_data.append({
                    'entity': entity_name, 'dataset': dataset_name,
                    'method': COORDINATED_ASSESSMENT_METHOD_IDENTIFIER,
                    'score': score, 'error_below': error_below, 'error_above': error_above, 'n_samples': n_samples_ci
                })

            # --- LLM Handling ---
            else:
                for llm_method in llm_methods_to_load:
                    cache_filename = f"CI_{entity_name}_{dataset_name}_{llm_method}_n{n_val_param}_w{wait_val_param}.json"
                    fpath_cache = os.path.join(CI_CACHE_DIR, cache_filename)
                    
                    score, error_below, error_above, n_samples_ci = None, 0, 0, 0

                    if os.path.exists(fpath_cache):
                        try:
                            with open(fpath_cache, 'r') as cf:
                                cached_data = json.load(cf)
                            score = cached_data['score']
                            error_below = cached_data['error_below']
                            error_above = cached_data['error_above']
                            n_samples_ci = cached_data['n_samples']
                            print(f"  CACHE HIT: {entity_name} ({llm_method}) on {dataset_name}. Score: {score if score is not None else 'N/A'}")
                            files_loaded_from_cache +=1
                        except Exception as e:
                            print(f"  CACHE ERROR (LLM - {entity_name}/{llm_method}/{dataset_name}): {e}. Recalculating.")
                            score = None # Force recalculation

                    if score is None: # Not in cache or cache error
                        filename_prefix = "AI_llama" if "llama" in entity_name.lower() else "AI_qwen"
                        fname_rationale = f"{filename_prefix}-{dataset_name}-{entity_name}-{llm_method}-n{n_val_param}-wait{wait_val_param}-rationales.jsonl"
                        fpath_rationale = os.path.join(responses_dir, fname_rationale)

                        if os.path.exists(fpath_rationale):
                            try:
                                with open(fpath_rationale, 'r') as f:
                                    rationales = json.load(f)
                                if not isinstance(rationales, list):
                                    print(f"    ERROR: LLM rationale file {fname_rationale} not a JSON list.")
                                    rationales = [] ; files_error_processing +=1
                                
                                score, error_below, error_above, n_samples_ci = calculate_bootstrap_ci_and_error_values(rationales)
                                if score is not None:
                                    print(f"  BOOTSTRAPPED: {entity_name} ({llm_method}) on {dataset_name}. Score: {score:.2f} (N={n_samples_ci}), Err: [{error_below:.2f}, {error_above:.2f}]")
                                    with open(fpath_cache, 'w') as cf:
                                        json.dump({'score': score, 'error_below': error_below, 'error_above': error_above, 'n_samples': n_samples_ci}, cf)
                                    files_processed_bootstrap +=1
                                else:
                                    print(f"    WARN: Could not calculate F1/CI for {entity_name} ({llm_method}) on {dataset_name} from {fname_rationale}.")
                                    files_error_processing +=1
                            except Exception as e:
                                print(f"    ERROR processing LLM file {fname_rationale}: {e}")
                                files_error_processing +=1
                        else:
                            print(f"    Info: LLM rationale file NOT FOUND: {fpath_rationale}")
                            files_missing_rationales +=1
                    
                    all_data.append({
                        'entity': entity_name, 'dataset': dataset_name, 
                        'method': llm_method, 'score': score,
                        'error_below': error_below, 'error_above': error_above, 'n_samples': n_samples_ci
                    })
    
    print(f"\nFinished loading/calculating scores. Cache Hits: {files_loaded_from_cache}, Bootstrapped Fresh: {files_processed_bootstrap}, Rationale Files Missing: {files_missing_rationales}, Processing Errors: {files_error_processing}")
    if not all_data:
        print("Warning: No data was loaded. Cannot generate plot.", file=sys.stderr)
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
    df['error_below'] = pd.to_numeric(df['error_below'], errors='coerce').fillna(0)
    df['error_above'] = pd.to_numeric(df['error_above'], errors='coerce').fillna(0)
    df['n_samples'] = pd.to_numeric(df['n_samples'], errors='coerce').fillna(0)
    
    df['model_family'] = df['entity'].map(MODEL_FAMILY_MAP) # Kept for consistency
    df['model_size'] = df['entity'].map(MODEL_SIZE_MAP)   # Kept for consistency
    
    df = df.sort_values(by=['dataset', 'entity', 'method'])
    return df

# --- Load Data ---
scores_df = load_data_for_plotting(
    RESPONSES_DIR,
    X_AXIS_ENTITIES_INTERNAL,
    ALLOWED_DATASETS,
    LLM_METHODS_INTERNAL,
    N_VAL,
    WAIT_VAL
)

# --- Validate Data ---
if scores_df.empty:
    print("Exiting: No data available to plot (DataFrame is empty). Check loading warnings.")
    if len(X_AXIS_ENTITIES_INTERNAL) > 0 and len(ALLOWED_DATASETS) > 0 :
        sys.exit(1)
elif not scores_df.empty and (scores_df['score'].isnull().all() or (scores_df['score'] == 0).all()):
    print("Warning: All loaded scores are effectively zero or NaN. Plot may show only zero-height bars.")

# --- Plot Generation ---
print("\n--- Generating Plot ---")
if scores_df.empty and (len(X_AXIS_ENTITIES_INTERNAL) == 0 or len(ALLOWED_DATASETS) == 0):
    print("Skipping plot generation as there are no entities or datasets to plot.")
else:
    num_rows = 1
    num_cols = len(ALLOWED_DATASETS)

    if num_cols == 0 and not scores_df.empty: # Only error if we had data but no datasets to plot on
        print("Error: No datasets defined for plotting. Exiting.")
        sys.exit(1)
    elif num_cols == 0: # No data and no datasets
        print("No datasets to plot.")


    if num_cols > 0 : # Proceed with plotting only if there are datasets
        fig, axes = plt.subplots(num_rows, num_cols,
                                figsize=FIG_SIZE,
                                sharey=True,
                                squeeze=False)

        final_legend_handles = []
        final_legend_labels = []
        llm_legend_items_gathered = False

        for j_dataset_idx, dataset_short_name in enumerate(ALLOWED_DATASETS):
            ax = axes[0, j_dataset_idx]
            dataset_subplot_data = scores_df[scores_df['dataset'] == dataset_short_name].copy()
            dataset_display_name = DATASET_DISPLAY_MAP.get(dataset_short_name, dataset_short_name.upper())
            ax.set_title(dataset_display_name, fontsize=TITLE_FONTSIZE)
            
            x_positions = np.array(CUSTOM_X_AXIS_POSITIONS)
            if len(x_positions) != len(X_AXIS_ENTITIES_INTERNAL):
                print("Error: CUSTOM_X_AXIS_POSITIONS length mismatch. Using default.")
                x_positions = np.arange(len(X_AXIS_ENTITIES_INTERNAL))
            
            for i_entity_idx, entity_internal_name in enumerate(X_AXIS_ENTITIES_INTERNAL):
                current_x_center = x_positions[i_entity_idx]
                entity_data = dataset_subplot_data[dataset_subplot_data['entity'] == entity_internal_name]

                if entity_data.empty:
                    print(f"    No data for entity '{entity_internal_name}' in dataset '{dataset_short_name}'. Skipping.")
                    continue
                
                if entity_internal_name == "CoDE":
                    code_entry = entity_data[entity_data['method'] == COORDINATED_ASSESSMENT_METHOD_IDENTIFIER]
                    if not code_entry.empty:
                        code_score = code_entry['score'].iloc[0] if pd.notna(code_entry['score'].iloc[0]) else 0.0
                        err_b = code_entry['error_below'].iloc[0] if pd.notna(code_entry['error_below'].iloc[0]) else 0.0
                        err_a = code_entry['error_above'].iloc[0] if pd.notna(code_entry['error_above'].iloc[0]) else 0.0
                        yerr_values = np.array([[err_b], [err_a]])
                        ax.bar(current_x_center, code_score, BAR_WIDTH, yerr=yerr_values, 
                                capsize=CAPSIZE_FOR_ERROR_BARS, color=CODE_BAR_COLOR, 
                                edgecolor=BAR_EDGE_COLOR, hatch=CODE_BAR_HATCH)
                else: # LLM
                    num_llm_methods = len(LLM_METHODS_INTERNAL)
                    group_width = BAR_WIDTH * num_llm_methods
                    start_offset = - (group_width / 2) + (BAR_WIDTH / 2)

                    baseline_s_entry = entity_data[entity_data['method'] == BASELINE_LLM_METHOD]['score']
                    llm_baseline_score = baseline_s_entry.iloc[0] if not baseline_s_entry.empty and pd.notna(baseline_s_entry.iloc[0]) else 0.0

                    for k_method_idx, llm_method_key in enumerate(LLM_METHODS_INTERNAL):
                        method_data = entity_data[entity_data['method'] == llm_method_key]
                        if not method_data.empty:
                            score = method_data['score'].iloc[0] if pd.notna(method_data['score'].iloc[0]) else 0.0
                            err_b = method_data['error_below'].iloc[0] if pd.notna(method_data['error_below'].iloc[0]) else 0.0
                            err_a = method_data['error_above'].iloc[0] if pd.notna(method_data['error_above'].iloc[0]) else 0.0
                            yerr_values = np.array([[err_b], [err_a]])
                            
                            bar_x_pos = current_x_center + start_offset + k_method_idx * BAR_WIDTH
                            legend_lbl = LLM_METHOD_NAME_MAPPING.get(llm_method_key, llm_method_key)
                            
                            rects = ax.bar(bar_x_pos, score, BAR_WIDTH, yerr=yerr_values, 
                                            capsize=CAPSIZE_FOR_ERROR_BARS, label=legend_lbl,
                                            color=COLORBLIND_FRIENDLY_PALETTE[k_method_idx % len(COLORBLIND_FRIENDLY_PALETTE)],
                                            edgecolor=BAR_EDGE_COLOR)
                            
                            if not llm_legend_items_gathered and legend_lbl not in final_legend_labels:
                                final_legend_handles.append(rects[0])
                                final_legend_labels.append(legend_lbl)

                            # Display relative difference ONLY for specific methods compared to the baseline
                            if llm_method_key in METHODS_TO_ANNOTATE_DIFF and not (score == 0 and llm_baseline_score == 0):
                                diff_text = ""
                                if llm_baseline_score != 0: # Avoid division by zero
                                    rel_diff = ((score - llm_baseline_score) / llm_baseline_score) * 100
                                    diff_text = f"+{rel_diff:.0f}%" if rel_diff >= 0 else f"{rel_diff:.0f}%"
                                elif score > 0: # Baseline is 0, current score is positive
                                    diff_text = "N/A" # Or some other indicator for infinite improvement
                                # If score is 0 and baseline is 0, diff_text remains "" (no change)
                                
                                if diff_text:
                                    text_y = max(score, 0) + err_a + 2 
                                    ax.text(bar_x_pos, text_y, diff_text, ha='center', va='bottom', 
                                            fontsize=ANNOTATION_FONTSIZE, color=ANNOTATION_COLOR)
                if final_legend_handles and not llm_legend_items_gathered:
                        llm_legend_items_gathered = True

            if j_dataset_idx == 0:
                ax.set_ylabel('Macro F1 Score (%)', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([ENTITY_PLOT_DISPLAY_NAME_MAP.get(e, e) for e in X_AXIS_ENTITIES_INTERNAL],
                                fontsize=TICK_LABEL_FONTSIZE, rotation=0)
            ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE, width=TICK_WIDTH, length=TICK_LENGTH)
            ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE, width=TICK_WIDTH, length=TICK_LENGTH)
            ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=GRID_LINEWIDTH)
            ax.set_ylim(0, 105)

        code_patch = mpatches.Patch(facecolor=CODE_BAR_COLOR, edgecolor=BAR_EDGE_COLOR, hatch=CODE_BAR_HATCH, label='CoDE')
        final_legend_handles.append(code_patch)
        final_legend_labels.append(ENTITY_PLOT_DISPLAY_NAME_MAP.get("CoDE", "CoDE") + ' (trained on D3)')

        if final_legend_handles:
            fig.legend(final_legend_handles, final_legend_labels, loc='upper center', 
                        ncol=len(final_legend_labels), bbox_to_anchor=(0.5, 1.0), fontsize=LEGEND_FONTSIZE)
        plt.tight_layout(rect=[0, 0, 1, 0.90]) # Adjust for legend
        
        # Ensure the PLOTS_DIR from config exists
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True) # <--- ADD DIRECTORY CREATION

        output_filename_only = "ai_generation_macro_f1_bootstrap_ci_plot.png" # Keep the base filename
        output_filepath = config.PLOTS_DIR / output_filename_only # <--- CONSTRUCT FULL PATH

        plt.savefig(output_filepath, dpi=300, bbox_inches='tight') # <--- USE output_filepath
        print(f"--- Plot Generation Complete --- Plot saved to {output_filepath}") # <--- USE output_filepath
    else: # num_cols == 0
        print("Plot generation skipped as no datasets were available to plot.")

