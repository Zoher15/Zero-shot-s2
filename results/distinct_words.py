import sys
from pathlib import Path
import logging
import re
import math
import string
from collections import Counter, defaultdict
import pickle
import colorsys

import nltk # Keep for WordNetLemmatizer, word_tokenize
from nltk.stem import WordNetLemmatizer
# stopwords will be accessed via helpers.preprocess_text_to_token_set

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm.auto import tqdm # For progress bars
import numpy as np

# --- Project Setup ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import config
from utils import helpers # Import your helpers module

# --- Logger Setup ---
helpers.setup_global_logger(config.RESULTS_DISTINCT_WORDS_LOG_FILE)
# Get a logger instance for this specific module.
logger = logging.getLogger(__name__)

# --- NLTK Resource Check (using helper) ---
# This ensures resources are available before WordNetLemmatizer() is called or stopwords are used.
# The helper function itself uses logging now.
REQUIRED_NLTK_RESOURCES = [
    ('corpora/wordnet', 'wordnet'),
    ('corpora/stopwords', 'stopwords'),
    ('tokenizers/punkt', 'punkt')
    # 'punkt_tab/english.pickle' was specific to an old local check, punkt should cover it.
]
helpers.ensure_nltk_resources(REQUIRED_NLTK_RESOURCES)

# Initialize global lemmatizer (after NLTK resources are checked)
try:
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    logger.error(f"Failed to initialize WordNetLemmatizer, NLTK resources might still be an issue: {e}", exc_info=True)
    sys.exit(1)


# --- Matplotlib Backend ---
try:
    plt.switch_backend('Agg')
    logger.info("Using Matplotlib backend: Agg")
except ImportError:
    logger.warning("Agg backend not available for Matplotlib, using default.")


# --- Configuration (from original script, some could move to config.py if more global) ---
# Paths from config.py are used directly where appropriate.

# Models, Methods, and Datasets (specific to this analysis)
MODELS_ABBR_DISTINCT = ['llama3.2'] # As per original distinct_words.py
MODEL_NAME_MAP_FULL_DISTINCT = {'qwen2.5': 'qwen25-7b', 'llama3.2': 'llama3-11b'} # As per original
METHODS_DISTINCT = ['zeroshot', 'zeroshot-cot', 'zeroshot-2-artifacts'] # As per original
DATASETS_DISTINCT = ['d3', 'df40', 'genimage'] # As per original

# Plot Configuration
MAX_WORDS_CLOUD_DISTINCT = 50
MAX_WORDS_BAR_DISTINCT = 50
GENERATE_WORDCLOUDS_DISTINCT = False # Set to False to turn off word cloud generation

FIGURE_WIDTH_PER_COLUMN_DISTINCT = 2.5
FIGURE_HEIGHT_PER_WC_ROW_DISTINCT = 0 # If wordclouds off, this is not used for height calculation
FIGURE_HEIGHT_PER_BAR_ROW_DISTINCT = 7
METHOD_LABEL_FONTSIZE_DISTINCT = 10
SUBPLOT_TITLE_FONTSIZE_DISTINCT = 10
BAR_PLOT_YLABEL_FONTSIZE_DISTINCT = 7
BAR_PLOT_XLABEL_FONTSIZE_DISTINCT = 7
BAR_PLOT_PERCENTAGE_FONTSIZE_DISTINCT = 7

# Manual Skip Words List (can be expanded or moved to config)
MANUAL_SKIP_WORDS_DISTINCT = set([])

# Method name mapping for titles/labels
METHOD_DISPLAY_NAME_MAPPING_DISTINCT = {
    'zeroshot': 'zero-shot',
    'zeroshot-cot': 'zero-shot-cot',
    'zeroshot-2-artifacts': r'zero-shot-s$^2$'
}

# Color Palette and Mapping (uses helpers.COLORBLIND_FRIENDLY_PALETTE if that's made global, or define locally)
# Assuming a COLORBLIND_FRIENDLY_PALETTE exists in helpers or config, or define it here.
# For now, using the one defined in the original distinct_words.py
COLORBLIND_FRIENDLY_PALETTE_DISTINCT = {
    'zeroshot': "#2A9D8F",
    'zeroshot-cot': "#E76F51",
    'zeroshot-2-artifacts': "#7F4CA5"
}


# --- Core Logic Functions (Log Odds, Plotting - with logging) ---

def extract_texts_from_rationale_records(rationale_records: list, text_field_key: str = 'rationales') -> list:
    """
    Extracts text strings for analysis from a list of rationale records.
    A rationale record is a dict, and the text might be in a field which itself
    could be a string or a list of strings.
    """
    extracted_texts = []
    if not rationale_records:
        return extracted_texts
    for record in rationale_records:
        if not isinstance(record, dict):
            # logger.warning(f"Skipping non-dictionary record during text extraction: {str(record)[:100]}")
            continue
        
        content_to_analyze = record.get(text_field_key) # Defaulting to 'rationales' key
        
        if isinstance(content_to_analyze, str):
            if content_to_analyze.strip():
                extracted_texts.append(content_to_analyze)
        elif isinstance(content_to_analyze, list):
            for text_item in content_to_analyze:
                if isinstance(text_item, str) and text_item.strip():
                    extracted_texts.append(text_item)
        # else:
            # logger.debug(f"No suitable text found in record under key '{text_field_key}': {str(record)[:100]}")
    return extracted_texts


def logodds(corpora_dic: dict, bg_counter: Counter) -> dict: # From original
    corp_size = {name: sum(counter.values()) for name, counter in corpora_dic.items()}
    bg_size = sum(bg_counter.values())
    result = {name: {} for name in corpora_dic}
    all_words = set()

    # Use MANUAL_SKIP_WORDS_DISTINCT here
    for counter_val in corpora_dic.values():
        all_words.update(word for word in counter_val if word not in MANUAL_SKIP_WORDS_DISTINCT)
    all_words.update(word for word in bg_counter if word not in MANUAL_SKIP_WORDS_DISTINCT)

    if not all_words:
        logger.warning("No words found (after skipping manual words) to calculate log odds.")
        return result
    
    logger.info(f"Calculating log odds for {len(corpora_dic)} corpora ({len(all_words)} unique words considered)...")
    
    # Alpha_0 calculation from the original script
    alpha_0 = bg_size / len(bg_counter) if bg_counter and len(bg_counter) > 0 else 1.0
    alpha_0 = max(alpha_0, 0.01) # Ensure alpha_0 is not too small

    for name, c_corpus in tqdm(corpora_dic.items(), desc="Calculating Log Odds", leave=False):
        ni_val = corp_size.get(name, 0)
        # nj calculation was: sum(size for other_name, size in corp_size.items() if other_name != name)
        # For current logodds (vs rest), comparison_counter is more direct for fj
        if ni_val == 0:
            logger.debug(f"Skipping log odds for '{name}' as its corpus size (ni) is 0.")
            continue

        comparison_counter_logodds = Counter()
        for other_name_logodds, other_counter_logodds in corpora_dic.items():
            if other_name_logodds != name:
                comparison_counter_logodds.update(other_counter_logodds)
        
        nj_val = sum(comparison_counter_logodds.values()) # Total words in all other corpora

        if nj_val == 0: # If there are no other corpora to compare against or they are empty
            logger.debug(f"Skipping log odds for '{name}' as comparison corpus size (nj) is 0.")
            continue


        for word_logodds in all_words:
            fi_val = c_corpus.get(word_logodds, 0)
            fj_val = comparison_counter_logodds.get(word_logodds, 0)

            # Using the prior calculation from the original script
            prior_i_val = alpha_0 / len(all_words) if all_words else 0.01
            prior_j_val = alpha_0 / len(all_words) if all_words else 0.01

            fi_smoothed_val = fi_val + prior_i_val
            ni_smoothed_val = ni_val + alpha_0 # Sum of priors for corpus i
            
            fj_smoothed_val = fj_val + prior_j_val
            nj_smoothed_val = nj_val + alpha_0 # Sum of priors for corpus j (all others)


            if not (fi_smoothed_val > 0 and (ni_smoothed_val - fi_smoothed_val) > 0 and \
                    fj_smoothed_val > 0 and (nj_smoothed_val - fj_smoothed_val) > 0):
                continue # Avoid math errors with log(0) or log(negative)

            try:
                log_odds_ratio_val = (math.log(fi_smoothed_val) - math.log(ni_smoothed_val - fi_smoothed_val)) - \
                                     (math.log(fj_smoothed_val) - math.log(nj_smoothed_val - fj_smoothed_val))
                
                variance_val = (1.0 / fi_smoothed_val) + (1.0 / (ni_smoothed_val - fi_smoothed_val)) + \
                               (1.0 / fj_smoothed_val) + (1.0 / (nj_smoothed_val - fj_smoothed_val))
                
                if variance_val <= 1e-9: # Avoid division by zero or tiny variance
                    continue
                std_dev_val = math.sqrt(variance_val)
                if std_dev_val > 1e-9: # Ensure std_dev is not effectively zero
                    result[name][word_logodds] = log_odds_ratio_val / std_dev_val
            except (ValueError, ZeroDivisionError) as e:
                # logger.debug(f"Math error calculating log odds for word '{word_logodds}' in corpus '{name}': {e}")
                continue
    return result


def generate_wordcloud_from_scores_local(word_scores_wc: dict, ax_wc, base_color_hex_wc: str): # Renamed
    positive_scores_wc = {word: score for word, score in word_scores_wc.items() if score > 0}
    if not positive_scores_wc:
        ax_wc.text(0.5, 0.5, "No distinctive words\n(score > 0)",
                   ha='center', va='center', transform=ax_wc.transAxes,
                   fontsize=9, color='grey', linespacing=1.5)
    else:
        # Use helper for color function
        color_func_wc = helpers.wordcloud_color_func_factory(base_color_hex_wc, positive_scores_wc)
        try:
            wc_img_h = 300
            wc_img_w = int(wc_img_h * (FIGURE_WIDTH_PER_COLUMN_DISTINCT / FIGURE_HEIGHT_PER_BAR_ROW_DISTINCT)) if FIGURE_HEIGHT_PER_BAR_ROW_DISTINCT > 0 else wc_img_h
            
            wc_obj = WordCloud(width=wc_img_w, height=wc_img_h,
                               background_color="white",
                               max_words=MAX_WORDS_CLOUD_DISTINCT,
                               max_font_size=60,
                               collocations=False,
                               color_func=color_func_wc, # Use helper's factory output
                               prefer_horizontal=0.9,
                               random_state=42
                              ).generate_from_frequencies(positive_scores_wc)
            ax_wc.imshow(wc_obj, interpolation="bilinear")
        except Exception as e:
            logger.error(f"Error generating word cloud with dims {wc_img_w}x{wc_img_h}: {e}", exc_info=True)
            ax_wc.text(0.5, 0.5, "WordCloud Error", ha='center', va='center', transform=ax_wc.transAxes, fontsize=9, color='red')
    ax_wc.set_xticks([])
    ax_wc.set_yticks([])
    ax_wc.axis("off")


def generate_response_percentage_barplot_local(top_word_scores_bar: dict, list_of_response_token_sets_bar: list, ax_bar, base_color_hex_bar: str): # Renamed
    word_percentages_bar = {}
    num_responses_bar = len(list_of_response_token_sets_bar)

    if num_responses_bar == 0:
        ax_bar.text(0.5, 0.5, "No responses found", ha='center', va='center', transform=ax_bar.transAxes, fontsize=BAR_PLOT_YLABEL_FONTSIZE_DISTINCT, color='grey')
        ax_bar.axis("off")
        return

    for word_bar, score_bar in top_word_scores_bar.items():
        occurrence_count_bar = sum(1 for response_token_set in list_of_response_token_sets_bar if word_bar in response_token_set)
        percentage_bar = (occurrence_count_bar / num_responses_bar) * 100 if num_responses_bar > 0 else 0
        word_percentages_bar[word_bar] = percentage_bar
    
    words_for_plot_bar = list(word_percentages_bar.keys())
    percentages_for_plot_bar = [word_percentages_bar[w] for w in words_for_plot_bar]

    if not words_for_plot_bar:
        ax_bar.text(0.5, 0.5, "No words to plot", ha='center', va='center', transform=ax_bar.transAxes, fontsize=BAR_PLOT_YLABEL_FONTSIZE_DISTINCT, color='grey')
        ax_bar.axis("off")
        return

    y_pos_bar = np.arange(len(words_for_plot_bar))
    ax_bar.barh(y_pos_bar, percentages_for_plot_bar, align='center', color=base_color_hex_bar)
    ax_bar.set_yticks(y_pos_bar)
    ax_bar.set_yticklabels(words_for_plot_bar, fontsize=BAR_PLOT_YLABEL_FONTSIZE_DISTINCT)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('% of Responses', fontsize=BAR_PLOT_XLABEL_FONTSIZE_DISTINCT)
    ax_bar.tick_params(axis='x', labelsize=BAR_PLOT_YLABEL_FONTSIZE_DISTINCT)
    ax_bar.tick_params(axis='y', labelsize=BAR_PLOT_YLABEL_FONTSIZE_DISTINCT) # Ensure y-tick labels are also small
    
    # Dynamic x-axis limit
    max_percentage_val = max(percentages_for_plot_bar) if percentages_for_plot_bar else 0
    ax_bar.set_xlim(0, max(100, max_percentage_val * 1.15 if max_percentage_val > 0 else 100))
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)


# --- Main Execution ---
def main():
    logger.info("--- Distinct Words Script Execution Started ---")
    model_corpora_main = defaultdict(lambda: defaultdict(Counter))
    model_bg_counters_main = defaultdict(Counter)
    raw_responses_text_map = defaultdict(lambda: defaultdict(list)) # Stores extracted text strings
    individual_processed_token_sets = defaultdict(lambda: defaultdict(list)) # Stores sets of tokens per response

    agg_data_loaded_flag = False
    individual_data_loaded_flag = False

    # Try to load aggregate data
    if config.PROCESSED_AGGREGATE_DATA_PKL.exists():
        logger.info(f"Loading aggregate processed data from: {config.PROCESSED_AGGREGATE_DATA_PKL}")
        try:
            with open(config.PROCESSED_AGGREGATE_DATA_PKL, 'rb') as f: loaded_agg_data = pickle.load(f)
            if isinstance(loaded_agg_data, dict) and \
               'corpora' in loaded_agg_data and 'bg_counters' in loaded_agg_data and \
               'raw_responses_texts' in loaded_agg_data: # Check for new key if saving texts
                temp_corpora_load = loaded_agg_data['corpora']
                for model_key, methods_data_load in temp_corpora_load.items():
                    model_corpora_main[model_key] = defaultdict(Counter, methods_data_load)
                model_bg_counters_main = defaultdict(Counter, loaded_agg_data['bg_counters'])
                # Load raw texts if saved this way
                temp_raw_texts = loaded_agg_data['raw_responses_texts']
                for model_key, methods_data_texts in temp_raw_texts.items():
                    raw_responses_text_map[model_key] = defaultdict(list, methods_data_texts)

                logger.info("Successfully loaded aggregate processed data (corpora, bg_counters, raw_texts).")
                agg_data_loaded_flag = True # This part is loaded
            else:
                logger.warning("Aggregate data PKL has unexpected structure. Reprocessing needed.")
        except Exception as e:
            logger.error(f"Error loading aggregate data from PKL: {e}. Reprocessing.", exc_info=True)

    # Try to load individual cleaned responses data (sets of tokens)
    if config.PROCESSED_INDIVIDUAL_RESPONSES_PKL.exists():
        logger.info(f"Loading individual cleaned responses data from: {config.PROCESSED_INDIVIDUAL_RESPONSES_PKL}")
        try:
            with open(config.PROCESSED_INDIVIDUAL_RESPONSES_PKL, 'rb') as f: loaded_ind_data = pickle.load(f)
            if isinstance(loaded_ind_data, dict):
                for model_k, methods_d in loaded_ind_data.items():
                    individual_processed_token_sets[model_k] = defaultdict(list)
                    for method_k, list_of_token_lists in methods_d.items():
                         # PKL stores lists of tokens, convert back to sets of tokens
                        individual_processed_token_sets[model_k][method_k] = [set(s) for s in list_of_token_lists]
                logger.info("Successfully loaded individual cleaned responses data (token sets).")
                individual_data_loaded_flag = True
            else:
                logger.warning("Individual responses PKL has unexpected structure. Reprocessing needed.")
        except Exception as e:
            logger.error(f"Error loading individual responses PKL: {e}. Reprocessing.", exc_info=True)


    if not agg_data_loaded_flag or not individual_data_loaded_flag:
        logger.info("One or both processed data files not found/invalid. Processing responses from scratch...")
        # Reset data structures if reprocessing
        model_corpora_main = defaultdict(lambda: defaultdict(Counter))
        model_bg_counters_main = defaultdict(Counter)
        raw_responses_text_map = defaultdict(lambda: defaultdict(list))
        individual_processed_token_sets = defaultdict(lambda: defaultdict(list))

        if not config.RESPONSES_DIR.is_dir():
            logger.error(f"Responses directory '{config.RESPONSES_DIR}' not found. Cannot process data.")
            sys.exit(1) # Critical error

        for model_abbr_proc in tqdm(MODELS_ABBR_DISTINCT, desc="Models"):
            model_full_proc = MODEL_NAME_MAP_FULL_DISTINCT.get(model_abbr_proc)
            if not model_full_proc:
                logger.warning(f"No full model name for '{model_abbr_proc}'. Skipping.")
                continue
            
            current_model_bg_counter_proc = Counter()
            logger.info(f"Processing Model: {model_abbr_proc}")
            for method_proc in tqdm(METHODS_DISTINCT, desc=f"Methods for {model_abbr_proc}", leave=False):
                all_method_raw_texts_for_model = []
                for dataset_proc in DATASETS_DISTINCT:
                    # Construct filename (adjust prefix logic if needed, e.g. 'AI_llama' vs 'AI_qwen')
                    prefix_fname = "AI_llama" if "llama" in model_abbr_proc.lower() else "AI_qwen"
                    # Original distinct_words used "AI_util" for llama3.2, "AI_dev" for others.
                    # This needs to align with how eval scripts save. Assuming a simpler rule for now.
                    if model_abbr_proc == 'llama3.2': prefix_fname = "AI_llama" # Or "AI_util" if that's how files are named
                    
                    # Ensure -wait0 is part of filename IF your eval scripts save it that way when wait is 0.
                    # The refactored eval scripts likely omit -wait0.
                    # fname_rationale_proc = f"{prefix_fname}-{dataset_proc}-{model_full_proc}-{method_proc}-n1-wait0-rationales.jsonl"
                    fname_rationale_proc = f"{prefix_fname}-{dataset_proc}-{model_full_proc}-{method_proc}-n1-rationales.jsonl"
                    fpath_rationale_proc = config.RESPONSES_DIR / fname_rationale_proc

                    if not fpath_rationale_proc.exists():
                        logger.warning(f"Rationale file not found: {fpath_rationale_proc}")
                        continue
                    
                    rationale_records_list = helpers.load_rationales_from_file(fpath_rationale_proc)
                    # Specify which field in rationale_records_list contains the text for word analysis.
                    # Assuming it's in a field named 'rationales' which could be str or list of str.
                    texts_from_file = extract_texts_from_rationale_records(rationale_records_list, text_field_key='rationales')
                    all_method_raw_texts_for_model.extend(texts_from_file)
                
                raw_responses_text_map[model_abbr_proc][method_proc].extend(all_method_raw_texts_for_model)
                if not all_method_raw_texts_for_model:
                    logger.info(f"  No text responses extracted for method '{method_proc}'. Skipping token processing.")
                    continue
                
                logger.info(f"  Method '{method_proc}': Processing {len(all_method_raw_texts_for_model)} raw text responses.")
                for text_response_item in tqdm(all_method_raw_texts_for_model, desc=f"Tokenizing for {method_proc}", leave=False, mininterval=1.0):
                    # Preprocess for counters (less strict, for word frequencies)
                    tokens_for_counter = []
                    try: # Simplified preprocessing for counter
                        temp_stop_words = set(stopwords.words('english'))
                        temp_combined_skip = temp_stop_words.union(MANUAL_SKIP_WORDS_DISTINCT)
                        text_to_count = str(text_response_item) # Ensure string
                        text_to_count = re.sub(r'http\S+|www\S+|https\S+', '', text_to_count, flags=re.MULTILINE)
                        text_to_count = ' '.join(text_to_count.split())
                        temp_translator = str.maketrans('', '', string.punctuation + '’‘“”')
                        raw_toks = word_tokenize(text_to_count.lower())
                        for rt_item in raw_toks:
                            rt_cleaned_item = rt_item.translate(temp_translator)
                            if rt_cleaned_item.isalpha() and rt_cleaned_item not in temp_combined_skip:
                                lemma_val_item = lemmatizer.lemmatize(rt_cleaned_item) # Use global lemmatizer
                                if lemma_val_item and len(lemma_val_item) > 1 and lemma_val_item not in temp_combined_skip:
                                    tokens_for_counter.append(lemma_val_item)
                    except Exception as e_counter_proc:
                        logger.error(f"Error in simplified preprocessing for counter for text: '{str(text_to_count)[:30]}...': {e_counter_proc}", exc_info=True)
                    
                    if tokens_for_counter:
                        response_token_counts = Counter(tokens_for_counter)
                        model_corpora_main[model_abbr_proc][method_proc].update(response_token_counts)
                        current_model_bg_counter_proc.update(response_token_counts) # Update BG with same tokens

                    # Preprocess for individual response sets (more strict, for bar plots)
                    # Use the helper here:
                    processed_token_set = helpers.preprocess_text_to_token_set(
                        text_response_item, lemmatizer, MANUAL_SKIP_WORDS_DISTINCT
                    )
                    if processed_token_set:
                        individual_processed_token_sets[model_abbr_proc][method_proc].append(processed_token_set)
            
            model_bg_counters_main[model_abbr_proc] = current_model_bg_counter_proc

        # Save processed data
        logger.info(f"Saving aggregate processed data to: {config.PROCESSED_AGGREGATE_DATA_PKL}")
        try:
            save_corpora_dict = {k: dict(v) for k, v in model_corpora_main.items()}
            save_bg_counters_dict = dict(model_bg_counters_main)
            # Save the extracted raw texts instead of the original complex raw_responses_map
            save_raw_texts_dict = {k_model: {k_method: texts for k_method, texts in methods_data.items()}
                                   for k_model, methods_data in raw_responses_text_map.items()}
            save_agg_data_dict = {
                'corpora': save_corpora_dict,
                'bg_counters': save_bg_counters_dict,
                'raw_responses_texts': save_raw_texts_dict # Save extracted texts
            }
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(config.PROCESSED_AGGREGATE_DATA_PKL, 'wb') as f_agg: pickle.dump(save_agg_data_dict, f_agg)
            logger.info("Successfully saved aggregate processed data.")
        except Exception as e_save_agg:
            logger.error(f"Error saving aggregate processed data: {e_save_agg}", exc_info=True)

        logger.info(f"Saving individual cleaned responses data to: {config.PROCESSED_INDIVIDUAL_RESPONSES_PKL}")
        try:
            # Save list of lists of tokens (pickle can't save sets of sets directly in all desired structures)
            save_individual_data_list = {
                model_k_ind: {
                    method_k_ind: [list(token_s) for token_s in list_of_sets_ind]
                    for method_k_ind, list_of_sets_ind in methods_d_ind.items()
                }
                for model_k_ind, methods_d_ind in individual_processed_token_sets.items()
            }
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(config.PROCESSED_INDIVIDUAL_RESPONSES_PKL, 'wb') as f_ind: pickle.dump(save_individual_data_list, f_ind)
            logger.info("Successfully saved individual cleaned responses data.")
        except Exception as e_save_ind:
            logger.error(f"Error saving individual cleaned responses data: {e_save_ind}", exc_info=True)

    # --- Report Word Counts ---
    # (This section remains largely the same, using logger for output)
    logger.info("\n--- Processed Word Counts (from aggregate data) ---")
    # ... (use logger.info instead of print) ...

    # --- Calculate Log Odds Ratios ---
    all_log_odds_results_main = {}
    logger.info("Calculating log odds within each model family...")
    if not model_corpora_main or not model_bg_counters_main:
        logger.error("Cannot calculate log odds, aggregate processed data is missing or empty.")
        # sys.exit(1) # Decide if this is fatal
    else:
        calculation_successful_flag = True
        for model_abbr_logodds in MODELS_ABBR_DISTINCT:
            logger.info(f"--- Processing Model Family for Log Odds: {model_abbr_logodds} ---")
            current_corpora = model_corpora_main.get(model_abbr_logodds)
            current_bg = model_bg_counters_main.get(model_abbr_logodds)

            if not current_corpora or not any(current_corpora.values()):
                logger.warning(f"Corpora for model '{model_abbr_logodds}' missing/empty. Skipping log odds.")
                continue
            if not current_bg:
                logger.warning(f"Background counter for model '{model_abbr_logodds}' empty. Skipping log odds.")
                continue
            
            non_empty_method_corpora = {m: c for m, c in current_corpora.items() if c}
            if not non_empty_method_corpora:
                logger.warning(f"No non-empty method corpora for '{model_abbr_logodds}'. Skipping.")
                continue

            logger.info(f"Calculating log odds for {len(non_empty_method_corpora)} non-empty methods in {model_abbr_logodds}.")
            model_logodds_results = logodds(non_empty_method_corpora, current_bg) # Call local logodds

            if not model_logodds_results:
                logger.warning(f"Log odds calculation returned no results for model '{model_abbr_logodds}'.")
                calculation_successful_flag = False
                continue
            for method_res, scores_res in model_logodds_results.items():
                if scores_res: all_log_odds_results_main[f"{model_abbr_logodds}_{method_res}"] = scores_res
                else: logger.info(f"No log odds scores generated for {model_abbr_logodds}_{method_res}.")
            logger.info(f"--- Finished Log Odds for: {model_abbr_logodds} ---")
        
        if not all_log_odds_results_main and calculation_successful_flag:
             logger.warning("Log odds calculation produced no results for any model.")
        elif not all_log_odds_results_main and not calculation_successful_flag:
            logger.error("Log odds calculation produced no results AND failed for some models.")
        elif not calculation_successful_flag:
            logger.warning("Log odds calculation failed for one or more models. Proceeding with available results.")


    # --- Generate Combined Word Cloud and Bar Plots ---
    if not all_log_odds_results_main:
        logger.info("No log odds results to generate combined plots. Skipping visualization.")
    elif not individual_processed_token_sets and any(MODELS_ABBR_DISTINCT):
        logger.info("No individual cleaned response data for bar plots. Skipping visualization.")
    else:
        n_models_plot = len(MODELS_ABBR_DISTINCT)
        n_methods_plot = len(METHODS_DISTINCT)

        if n_models_plot == 0 or n_methods_plot == 0:
            logger.info("No models or methods defined to generate plots for.")
        else:
            rows_per_model_plot = 2 if GENERATE_WORDCLOUDS_DISTINCT else 1
            n_plot_rows_total = n_models_plot * rows_per_model_plot
            
            total_fig_h = 0
            if GENERATE_WORDCLOUDS_DISTINCT:
                total_fig_h += (FIGURE_HEIGHT_PER_WC_ROW_DISTINCT * n_models_plot)
            total_fig_h += (FIGURE_HEIGHT_PER_BAR_ROW_DISTINCT * n_models_plot)
            
            total_fig_w = FIGURE_WIDTH_PER_COLUMN_DISTINCT * n_methods_plot
            
            if total_fig_w <= 0 or total_fig_h <= 0 :
                 logger.error(f"Calculated figure dimensions are invalid: W={total_fig_w}, H={total_fig_h}. Skipping plot.")
            else:
                fig_plot, axes_plot = plt.subplots(nrows=n_plot_rows_total, ncols=n_methods_plot,
                                             figsize=(total_fig_w, total_fig_h),
                                             squeeze=False)
                logger.info("Generating combined plots for distinct words...")

                for r_model_idx_plot, model_abbr_plot in enumerate(MODELS_ABBR_DISTINCT):
                    for c_method_idx_plot, method_plot in enumerate(METHODS_DISTINCT):
                        corpus_key_plot = f"{model_abbr_plot}_{method_plot}"
                        method_log_odds_scores_plot = all_log_odds_results_main.get(corpus_key_plot, {})
                        base_color_plot = COLORBLIND_FRIENDLY_PALETTE_DISTINCT.get(method_plot, "#000000")
                        
                        current_row_offset_plot = r_model_idx_plot * rows_per_model_plot

                        if GENERATE_WORDCLOUDS_DISTINCT:
                            ax_wc_plot = axes_plot[current_row_offset_plot, c_method_idx_plot]
                            generate_wordcloud_from_scores_local(method_log_odds_scores_plot, ax_wc_plot, base_color_plot)
                            # No title on subplot for wc

                        bar_chart_row_idx_plot = current_row_offset_plot + (1 if GENERATE_WORDCLOUDS_DISTINCT else 0)
                        ax_bar_plot = axes_plot[bar_chart_row_idx_plot, c_method_idx_plot]
                        
                        sorted_scores_plot = sorted(method_log_odds_scores_plot.items(), key=lambda item: item[1], reverse=True)
                        top_n_words_bar_plot = dict(sorted_scores_plot[:MAX_WORDS_BAR_DISTINCT])
                        current_cleaned_responses_bar_plot = individual_processed_token_sets.get(model_abbr_plot, {}).get(method_plot, [])

                        if not top_n_words_bar_plot:
                            ax_bar_plot.text(0.5, 0.5, "No top words", ha='center', va='center', transform=ax_bar_plot.transAxes, fontsize=BAR_PLOT_YLABEL_FONTSIZE_DISTINCT, color='grey')
                            ax_bar_plot.axis("off")
                        elif not current_cleaned_responses_bar_plot:
                             ax_bar_plot.text(0.5, 0.5, "No cleaned responses", ha='center', va='center', transform=ax_bar_plot.transAxes, fontsize=BAR_PLOT_YLABEL_FONTSIZE_DISTINCT, color='grey')
                             ax_bar_plot.axis("off")
                        else:
                            generate_response_percentage_barplot_local(top_n_words_bar_plot, current_cleaned_responses_bar_plot, ax_bar_plot, base_color_plot)
                        
                        # No title on subplot for bar if wordclouds are off

                fig_plot.subplots_adjust(left=0, bottom=0.08 if n_methods_plot > 0 else 0.05, right=1, top=1, wspace=0.2, hspace=0.1 if GENERATE_WORDCLOUDS_DISTINCT else 0)

                if n_methods_plot > 0: # Add method labels at bottom
                    for c_idx_label, method_label in enumerate(METHODS_DISTINCT):
                        ax_for_label_pos = axes_plot[0, c_idx_label]
                        pos_label = ax_for_label_pos.get_position()
                        center_x_fig_coords = pos_label.x0 + pos_label.width / 2.0
                        y_pos_label_fig_coords = 0.0 # Bottom of the figure
                        display_method_label = METHOD_DISPLAY_NAME_MAPPING_DISTINCT.get(method_label, method_label)
                        fig_plot.text(center_x_fig_coords, y_pos_label_fig_coords, display_method_label,
                                 va='bottom', ha='center', # Adjusted va
                                 fontsize=METHOD_LABEL_FONTSIZE_DISTINCT, transform=fig_plot.transFigure) # Use figure transform

                config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
                model_prefix_for_filename = MODELS_ABBR_DISTINCT[0] if len(MODELS_ABBR_DISTINCT) == 1 else "multi_model"
                plot_type_filename_suffix = "combined" if GENERATE_WORDCLOUDS_DISTINCT else "barcharts_only"
                # Versioning in filename for easier tracking
                output_filename_base_plot = f"{model_prefix_for_filename}_{MAX_WORDS_CLOUD_DISTINCT}wc_{MAX_WORDS_BAR_DISTINCT}bar_{plot_type_filename_suffix}_v_refactored"
                
                output_filepath_plot_png = config.PLOTS_DIR / f"{output_filename_base_plot}.png"
                output_filepath_plot_pdf = config.PLOTS_DIR / f"{output_filename_base_plot}.pdf" # Example for PDF

                try:
                    plt.savefig(output_filepath_plot_png, dpi=300, bbox_inches='tight')
                    logger.info(f"Combined plot saved to {output_filepath_plot_png}")
                    # plt.savefig(output_filepath_plot_pdf, dpi=300, bbox_inches='tight') # Optionally save PDF
                    # logger.info(f"Combined plot saved to {output_filepath_plot_pdf}")
                except Exception as e_save_plot:
                    logger.error(f"Error saving combined plot figure: {e_save_plot}", exc_info=True)
                plt.close(fig_plot)
                logger.info("Combined plot figure closed.")

    logger.info("--- Distinct Words Script Execution Finished ---")

if __name__ == "__main__":
    main()