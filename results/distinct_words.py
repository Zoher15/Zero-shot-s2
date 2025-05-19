# Import necessary libraries
import os
import json
import re
import math
import string # For punctuation removal
from collections import Counter, defaultdict
import pickle # For saving/loading processed data
import colorsys # For color manipulation (HSL)
from pathlib import Path # <--- ADD if not already there (it is used by config.py)
import sys # <--- ADD

# Assuming config.py is in the project root (parent of 'results')
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
import config # <--- ADD

# NLTK setup for text processing
import nltk
# --- (Include the corrected NLTK resource checking blocks here) ---
print("--- Checking NLTK Resources ---")
# Define a helper function to check and download NLTK data
def check_nltk_resource(resource_id, download_name=None):
    if download_name is None:
        download_name = resource_id.split('/')[-1] # Use last part of path if no specific name
    try:
        # Use find for corpora/tokenizers, specific check for punkt_tab
        if resource_id.startswith('corpora/') or resource_id.startswith('tokenizers/'):
            nltk.data.find(resource_id)
        elif resource_id == 'punkt_tab/english.pickle':
            nltk.data.find('tokenizers/punkt_tab/english.pickle') # Check specific file
        else:
            nltk.data.find(resource_id) # Default find for others like wordnet
        print(f"NLTK resource '{download_name}' found.")
    except LookupError:
        print(f"NLTK resource '{download_name}' not found. Downloading...")
        # Use the download_name for nltk.download
        nltk.download(download_name, quiet=True)
        print(f"Downloaded '{download_name}'.")
    except Exception as e:
        print(f"An error occurred checking/downloading {download_name}: {e}")

# Check required NLTK resources
check_nltk_resource('corpora/wordnet', 'wordnet')
check_nltk_resource('corpora/stopwords', 'stopwords')
check_nltk_resource('tokenizers/punkt', 'punkt')
check_nltk_resource('punkt_tab/english.pickle', 'punkt_tab') # Check specific pickle file

print("--- NLTK Resource Check Complete ---")


from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize global lemmatizer
lemmatizer = WordNetLemmatizer()


# WordCloud and plotting libraries
import matplotlib.pyplot as plt
# Explicitly set a backend that works well in various environments
try:
    plt.switch_backend('Agg')
    print("Using Matplotlib backend: Agg")
except ImportError:
    print("Agg backend not available, using default.")


from wordcloud import WordCloud
from tqdm.auto import tqdm # For progress bars
import numpy as np # For score normalization

# --- Configuration ---

# Data Paths
# Data Paths
RESPONSES_DIR = config.RESPONSES_DIR # <--- Primary responses directory from config

# Update paths for processed data to use config.py variables
PROCESSED_AGGREGATE_DATA_PATH = config.PROCESSED_AGGREGATE_DATA_PKL
PROCESSED_INDIVIDUAL_RESPONSES_PATH = config.PROCESSED_INDIVIDUAL_RESPONSES_PKL


# Models, Methods, and Datasets
MODELS_ABBR = ['llama3.2'] 
MODEL_NAME_MAP_FULL = {'qwen2.5': 'qwen25-7b', 'llama3.2': 'llama3-11b'}
METHODS = ['zeroshot', 'zeroshot-cot', 'zeroshot-2-artifacts']
DATASETS = ['d3', 'df40', 'genimage']

# --- Plot Configuration ---
MAX_WORDS_CLOUD = 50 
MAX_WORDS_BAR = 50   
GENERATE_WORDCLOUDS = False # Set to False to turn off word cloud generation

# Figure and Font Sizes
FIGURE_WIDTH_PER_COLUMN = 2.5  
FIGURE_HEIGHT_PER_WC_ROW = 0 
FIGURE_HEIGHT_PER_BAR_ROW = 7 
METHOD_LABEL_FONTSIZE = 10
SUBPLOT_TITLE_FONTSIZE = 10 
BAR_PLOT_YLABEL_FONTSIZE = 7
BAR_PLOT_XLABEL_FONTSIZE = 7
BAR_PLOT_PERCENTAGE_FONTSIZE = 7


# --- Manual Skip Words List ---
MANUAL_SKIP_WORDS = set([])

# Method name mapping for titles/labels
METHOD_DISPLAY_NAME_MAPPING = {
    'zeroshot': 'zero-shot',
    'zeroshot-cot': 'zero-shot-cot',
    'zeroshot-2-artifacts': r'zero-shot-s$^2$'
}

# --- Color Palette and Mapping ---
COLORBLIND_FRIENDLY_PALETTE = {
    'zeroshot': "#2A9D8F",
    'zeroshot-cot': "#E76F51",
    'zeroshot-2-artifacts': "#7F4CA5"
}


# --- Helper Functions ---
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color format: {hex_color}")
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb_color[0]*255), int(rgb_color[1]*255), int(rgb_color[2]*255)
    )

def adjust_lightness(rgb_color, factor):
    try:
        h, l, s = colorsys.rgb_to_hls(*rgb_color)
        min_lightness = 0.15
        max_lightness = 0.85
        new_l = max_lightness - factor * (max_lightness - min_lightness)
        new_l = max(0.05, min(0.95, new_l))
        return colorsys.hls_to_rgb(h, new_l, s)
    except Exception as e:
        print(f"Error adjusting lightness for color {rgb_color} with factor {factor}: {e}")
        return rgb_color

def color_func_factory(base_color_hex, word_scores):
    try:
        base_color_rgb = hex_to_rgb(base_color_hex)
    except ValueError as e:
        print(f"Error converting base hex color {base_color_hex}: {e}. Using black.")
        base_color_rgb = (0.0, 0.0, 0.0)

    positive_scores = {word: score for word, score in word_scores.items() if score > 0}

    if not positive_scores:
        min_score, max_score, score_range = 0, 1, 1
    else:
        scores_values = list(positive_scores.values())
        min_score = min(scores_values)
        max_score = max(scores_values)
        score_range = max_score - min_score
        if score_range <= 1e-6:
            score_range = 1

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        score = word_scores.get(word, 0)
        if score <= 0:
            lightness_factor = 0.0
        else:
            clamped_score = max(min_score, min(score, max_score))
            lightness_factor = (clamped_score - min_score) / score_range
        adjusted_rgb = adjust_lightness(base_color_rgb, lightness_factor)
        return rgb_to_hex(adjusted_rgb)
    return color_func

def load_responses(model_abbr, model_full_name, method, dataset, base_dir):
    if model_abbr == 'llama3.2':
        file_prefix = "AI_util"
    else:
        file_prefix = "AI_dev"
    file_name = f"{file_prefix}-{dataset}-{model_full_name}-{method}-n1-wait0-rationales.jsonl"
    file_path = os.path.join(base_dir, file_name)
    responses = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}") 
        return responses
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        if not file_content.strip():
            return responses
        try:
            data = json.loads(file_content)
            if isinstance(data, list):
                data_items = data
            elif isinstance(data, dict):
                data_items = [data]
            else:
                print(f"Warning: Unexpected JSON structure in {file_path}")
                return responses
        except json.JSONDecodeError:
            data_items = []
            file_content = file_content.strip()
            for line in file_content.splitlines():
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    data_items.append(item)
                except json.JSONDecodeError as e_line:
                    print(f"Error: JSON decode error on line in {file_path}: {e_line}. Line: '{line[:100]}...'")
                    continue
        for item in data_items:
            if isinstance(item, dict) and 'rationales' in item:
                rationales_list = item['rationales']
                if isinstance(rationales_list, list):
                    for rationale_text in rationales_list:
                        if isinstance(rationale_text, str) and rationale_text.strip():
                            responses.append(rationale_text)
                elif isinstance(rationales_list, str) and rationales_list.strip():
                    responses.append(rationales_list)
    except Exception as e:
        print(f"Error loading/processing file {file_path}: {type(e).__name__} - {e}")
    return responses

def preprocess_text_to_token_set(text, current_lemmatizer):
    """ Preprocesses text and returns a set of unique lemmatized tokens. """
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        print("Error: NLTK stopwords not found. Attempting download.")
        check_nltk_resource('corpora/stopwords', 'stopwords')
        stop_words = set(stopwords.words('english'))

    combined_skip_words = stop_words.union(MANUAL_SKIP_WORDS)
    processed_token_set = set() 
    try:
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = ' '.join(text.split())
        translator = str.maketrans('', '', string.punctuation + '’‘“”')
        tokens = word_tokenize(text.lower())
        for token in tokens:
            token = token.translate(translator)
            if token.isalpha() and token not in combined_skip_words:
                try:
                    lemma = current_lemmatizer.lemmatize(token)
                    if lemma and len(lemma) > 1 and lemma not in combined_skip_words:
                        processed_token_set.add(lemma) 
                except Exception as lem_e:
                    print(f"Error lemmatizing token '{token}': {lem_e}")
    except Exception as e:
        print(f"Error in preprocess_text_to_token_set for text starting with: '{str(text)[:50]}...' - Error: {e}")
        return set() 
    return processed_token_set

def logodds(corpora_dic, bg_counter):
    corp_size = {name: sum(counter.values()) for name, counter in corpora_dic.items()}
    bg_size = sum(bg_counter.values())
    result = {name: {} for name in corpora_dic}
    all_words = set()
    for counter in corpora_dic.values():
        all_words.update(word for word in counter if word not in MANUAL_SKIP_WORDS)
    all_words.update(word for word in bg_counter if word not in MANUAL_SKIP_WORDS)
    if not all_words:
        print("Warning: No words found (after skipping) to calculate log odds.")
        return result
    print(f"Calculating log odds for {len(corpora_dic)} corpora ({len(all_words)} unique words considered)...")
    alpha_0 = bg_size / len(bg_counter) if bg_counter and len(bg_counter) > 0 else 1.0
    alpha_0 = max(alpha_0, 0.01)

    for name, c in tqdm(corpora_dic.items(), desc="Calculating Log Odds", leave=False):
        ni = corp_size.get(name, 0)
        nj = sum(size for other_name, size in corp_size.items() if other_name != name)
        if ni == 0:
            continue
        comparison_counter = Counter()
        for other_name, other_counter in corpora_dic.items():
            if other_name != name:
                comparison_counter.update(other_counter)
        for word in all_words:
            fi = c.get(word, 0)
            fj = comparison_counter.get(word, 0)
            prior_i = alpha_0 / len(all_words) if all_words else 0.01
            prior_j = alpha_0 / len(all_words) if all_words else 0.01
            fi_smoothed = fi + prior_i
            ni_smoothed = ni + alpha_0
            fj_smoothed = fj + prior_j
            nj_smoothed = nj + alpha_0
            if fi_smoothed <= 0 or (ni_smoothed - fi_smoothed) <= 0 or \
               fj_smoothed <= 0 or (nj_smoothed - fj_smoothed) <= 0:
                continue
            try:
                log_odds_ratio = (math.log(fi_smoothed) - math.log(ni_smoothed - fi_smoothed)) - \
                                 (math.log(fj_smoothed) - math.log(nj_smoothed - fj_smoothed))
                variance = (1.0 / fi_smoothed) + (1.0 / (ni_smoothed - fi_smoothed)) + \
                           (1.0 / fj_smoothed) + (1.0 / (nj_smoothed - fj_smoothed))
                if variance <= 0:
                    continue
                std_dev = math.sqrt(variance)
                if std_dev > 1e-9:
                    result[name][word] = log_odds_ratio / std_dev
            except (ValueError, ZeroDivisionError):
                continue
    return result

def generate_wordcloud_from_scores(word_scores, ax, base_color_hex):
    positive_scores = {word: score for word, score in word_scores.items() if score > 0}
    if not positive_scores:
        ax.text(0.5, 0.5, "No distinctive words\n(score > 0)",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='grey', linespacing=1.5) 
    else:
        color_func = color_func_factory(base_color_hex, positive_scores)
        try:
            wc_img_height = 300
            if FIGURE_HEIGHT_PER_WC_ROW > 0: 
                 wc_img_width = int(wc_img_height * (FIGURE_WIDTH_PER_COLUMN / FIGURE_HEIGHT_PER_WC_ROW))
            else: 
                 wc_img_width = wc_img_height 

            wc = WordCloud(width=wc_img_width, height=wc_img_height, 
                           background_color="white",
                           max_words=MAX_WORDS_CLOUD, 
                           max_font_size=60, 
                           collocations=False,
                           color_func=color_func,
                           prefer_horizontal=0.9,
                           random_state=42
                          ).generate_from_frequencies(positive_scores)
            ax.imshow(wc, interpolation="bilinear")
        except Exception as e:
            print(f"Error generating word cloud with dims {wc_img_width}x{wc_img_height}: {e}")
            ax.text(0.5, 0.5, "WordCloud Error",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='red') 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

def generate_response_percentage_barplot_from_cleaned(top_word_scores, list_of_response_token_sets, ax, base_color_hex):
    word_percentages = {}
    num_responses = len(list_of_response_token_sets)

    if num_responses == 0:
        ax.text(0.5, 0.5, "No responses found", ha='center', va='center', transform=ax.transAxes, fontsize=BAR_PLOT_YLABEL_FONTSIZE, color='grey')
        ax.axis("off")
        return

    for word, score in top_word_scores.items(): 
        occurrence_count = 0
        for response_token_set in list_of_response_token_sets:
            if word in response_token_set: 
                occurrence_count += 1
        percentage = (occurrence_count / num_responses) * 100 if num_responses > 0 else 0
        word_percentages[word] = percentage
    
    words_for_plot = list(word_percentages.keys())
    percentages_for_plot = [word_percentages[w] for w in words_for_plot]

    if not words_for_plot:
        ax.text(0.5, 0.5, "No words to plot", ha='center', va='center', transform=ax.transAxes, fontsize=BAR_PLOT_YLABEL_FONTSIZE, color='grey')
        ax.axis("off")
        return

    y_pos = np.arange(len(words_for_plot))
    bars = ax.barh(y_pos, percentages_for_plot, align='center', color=base_color_hex)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words_for_plot, fontsize=BAR_PLOT_YLABEL_FONTSIZE) 
    ax.invert_yaxis()
    ax.set_xlabel('% of Responses', fontsize=BAR_PLOT_XLABEL_FONTSIZE) 
    ax.tick_params(axis='x', labelsize=BAR_PLOT_YLABEL_FONTSIZE) 
    ax.tick_params(axis='y', labelsize=BAR_PLOT_YLABEL_FONTSIZE)
    
    ax.set_xlim(0, max(100, max(percentages_for_plot) * 1.15 if percentages_for_plot else 100)) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# --- Main Execution ---

# 1. Load or Process Data
model_corpora = defaultdict(lambda: defaultdict(Counter)) 
model_bg_counters = defaultdict(Counter) 
raw_responses_map = defaultdict(lambda: defaultdict(list)) 
individual_processed_responses = defaultdict(lambda: defaultdict(list)) 

agg_data_loaded = False
individual_data_loaded = False

# Try to load aggregate data
if os.path.exists(PROCESSED_AGGREGATE_DATA_PATH):
    print(f"Loading aggregate processed data from: {PROCESSED_AGGREGATE_DATA_PATH}")
    try:
        with open(PROCESSED_AGGREGATE_DATA_PATH, 'rb') as f:
            loaded_data = pickle.load(f)
        if isinstance(loaded_data, dict) and \
           'corpora' in loaded_data and \
           'bg_counters' in loaded_data and \
           'raw_responses' in loaded_data:
            temp_corpora = loaded_data['corpora']
            for model, methods_data in temp_corpora.items():
                model_corpora[model] = defaultdict(Counter, methods_data)
            model_bg_counters = defaultdict(Counter, loaded_data['bg_counters'])
            temp_raw = loaded_data['raw_responses']
            for model, methods_data in temp_raw.items():
                raw_responses_map[model] = defaultdict(list, methods_data)
            print("Successfully loaded aggregate processed data.")
            agg_data_loaded = True
        else:
            print("Error: Aggregate data has unexpected structure. Reprocessing relevant parts...")
    except Exception as e:
        print(f"Error loading aggregate data: {e}. Reprocessing relevant parts...")

# Try to load individual cleaned responses data
if os.path.exists(PROCESSED_INDIVIDUAL_RESPONSES_PATH):
    print(f"Loading individual cleaned responses data from: {PROCESSED_INDIVIDUAL_RESPONSES_PATH}")
    try:
        with open(PROCESSED_INDIVIDUAL_RESPONSES_PATH, 'rb') as f:
            loaded_individual_data = pickle.load(f)
        if isinstance(loaded_individual_data, dict):
            for model_key, methods_data in loaded_individual_data.items():
                individual_processed_responses[model_key] = defaultdict(list)
                for method_key, list_of_item_lists in methods_data.items(): 
                    individual_processed_responses[model_key][method_key] = [set(s) for s in list_of_item_lists] 
            print("Successfully loaded individual cleaned responses data.")
            individual_data_loaded = True
        else:
            print("Error: Individual cleaned responses data has unexpected structure. Reprocessing...")
    except Exception as e:
        print(f"Error loading individual cleaned responses data: {e}. Reprocessing...")


# If any data is missing, reprocess
if not agg_data_loaded or not individual_data_loaded:
    print("One or both processed data files not found or invalid. Processing responses from scratch...")
    model_corpora = defaultdict(lambda: defaultdict(Counter))
    model_bg_counters = defaultdict(Counter)
    raw_responses_map = defaultdict(lambda: defaultdict(list))
    individual_processed_responses = defaultdict(lambda: defaultdict(list))

    if not os.path.isdir(RESPONSES_DIR):
        print(f"Error: Responses directory '{RESPONSES_DIR}' not found. Cannot process data.")
        if RESPONSES_DIR != './dummy_responses': exit()

    for model_abbr in tqdm(MODELS_ABBR, desc="Models"):
        model_full = MODEL_NAME_MAP_FULL.get(model_abbr)
        if not model_full:
            print(f"Warning: No full model name found for abbreviation '{model_abbr}'. Skipping.")
            continue
        current_model_bg_counter = Counter()
        print(f"\nProcessing Model: {model_abbr}") 
        for method in tqdm(METHODS, desc=f"Methods for {model_abbr}", leave=False):
            all_method_raw_responses = [] 
            for dataset in DATASETS:
                responses = load_responses(model_abbr, model_full, method, dataset, RESPONSES_DIR)
                all_method_raw_responses.extend(responses)
            raw_responses_map[model_abbr][method].extend(all_method_raw_responses)
            if not all_method_raw_responses:
                print(f"  No responses found for method '{method}'. Skipping token processing.")
                continue
            print(f"  Method '{method}': Loaded {len(all_method_raw_responses)} individual response corpora.")
            for text_response in tqdm(all_method_raw_responses, desc=f"Responses for {method}", leave=False, disable=True):
                temp_lemmatizer_for_counter = WordNetLemmatizer() 
                tokens_list_for_counter = []
                try:
                    stop_words_temp = set(stopwords.words('english'))
                    combined_skip_words_temp = stop_words_temp.union(MANUAL_SKIP_WORDS)
                    if not isinstance(text_response, str): text_response_str = str(text_response)
                    else: text_response_str = text_response
                    text_response_str = re.sub(r'http\S+|www\S+|https\S+', '', text_response_str, flags=re.MULTILINE)
                    text_response_str = ' '.join(text_response_str.split())
                    translator_temp = str.maketrans('', '', string.punctuation + '’‘“”')
                    raw_tokens = word_tokenize(text_response_str.lower())
                    for rt in raw_tokens:
                        rt_cleaned = rt.translate(translator_temp)
                        if rt_cleaned.isalpha() and rt_cleaned not in combined_skip_words_temp:
                            lemma_val = temp_lemmatizer_for_counter.lemmatize(rt_cleaned)
                            if lemma_val and len(lemma_val) > 1 and lemma_val not in combined_skip_words_temp:
                                tokens_list_for_counter.append(lemma_val)
                except Exception as e_proc:
                    print(f"Error in simplified preprocessing for counter for text: '{text_response_str[:30]}...': {e_proc}")
                if tokens_list_for_counter:
                    response_specific_counter = Counter(tokens_list_for_counter)
                    model_corpora[model_abbr][method].update(response_specific_counter)
                    current_model_bg_counter.update(tokens_list_for_counter)
                processed_token_set_for_response = preprocess_text_to_token_set(text_response, lemmatizer)
                if processed_token_set_for_response: 
                    individual_processed_responses[model_abbr][method].append(processed_token_set_for_response)
        model_bg_counters[model_abbr] = current_model_bg_counter

    # Save aggregate data
    print(f"\nSaving aggregate processed data to: {PROCESSED_AGGREGATE_DATA_PATH}")
    try:
        save_corpora = {k: dict(v) for k, v in model_corpora.items()}
        save_bg_counters = dict(model_bg_counters)
        save_raw_responses = {k_model: {k_method: list_responses for k_method, list_responses in methods_data.items()} 
                              for k_model, methods_data in raw_responses_map.items()}
        save_agg_data = {'corpora': save_corpora, 'bg_counters': save_bg_counters, 'raw_responses': save_raw_responses}
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True) # <--- ADD
        with open(PROCESSED_AGGREGATE_DATA_PATH, 'wb') as f: pickle.dump(save_agg_data, f)
        print("Successfully saved aggregate processed data.")
    except Exception as e: print(f"Error saving aggregate processed data: {e}")

    # Save individual cleaned responses data
    print(f"\nSaving individual cleaned responses data to: {PROCESSED_INDIVIDUAL_RESPONSES_PATH}")
    try:
        save_individual_data = {
            model_key: { 
                method_key: [list(token_set) for token_set in list_of_sets] 
                for method_key, list_of_sets in methods_data.items()
            }
            for model_key, methods_data in individual_processed_responses.items()
        }
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_INDIVIDUAL_RESPONSES_PATH, 'wb') as f: pickle.dump(save_individual_data, f)
        print("Successfully saved individual cleaned responses data.")
    except Exception as e: print(f"Error saving individual cleaned responses data: {e}")

# Report Word Counts
print("\n--- Processed Word Counts (from aggregate data) ---")
if not model_corpora:
    print("No aggregate data available to report counts.")
else:
    for model_abbr_key in MODELS_ABBR:
        if model_abbr_key in model_bg_counters and model_bg_counters[model_abbr_key]:
            total_family_words = sum(model_bg_counters[model_abbr_key].values())
            print(f"\nModel Family: {model_abbr_key}")
            print(f"  Total unique words (types) in background: {len(model_bg_counters[model_abbr_key]):,}")
            print(f"  Total words (tokens) in background (after filtering): {total_family_words:,}")
            if model_abbr_key in model_corpora:
                for method_key in METHODS:
                    num_raw_resp = len(raw_responses_map.get(model_abbr_key, {}).get(method_key, []))
                    num_cleaned_resp_sets = len(individual_processed_responses.get(model_abbr_key, {}).get(method_key, []))
                    print(f"    Method '{method_key}': {num_raw_resp} raw responses, {num_cleaned_resp_sets} cleaned response sets.")
                    method_counter = model_corpora[model_abbr_key].get(method_key)
                    if method_counter:
                        method_tokens = sum(method_counter.values())
                        method_types = len(method_counter)
                        print(f"      Aggregate Tokens: {method_tokens:,} tokens, {method_types:,} types")
                    else:
                        print(f"      Aggregate Tokens: No token data found")
            else:
                print("  No method-specific corpora found for this model.")
        else:
            print(f"\nModel Family: {model_abbr_key} - No data found or background counter is empty.")
print("---------------------------\n")

# 2. Calculate Log Odds Ratios
all_log_odds_results = {} 
print("Calculating log odds within each model family...")
if not model_corpora or not model_bg_counters:
    print("Error: Cannot calculate log odds, aggregate processed data is missing or empty.")
    if RESPONSES_DIR != './dummy_responses' and not agg_data_loaded : exit()
calculation_successful = True
for model_abbr in MODELS_ABBR:
    print(f"--- Processing Model Family for Log Odds: {model_abbr} ---")
    current_corpora_for_model = model_corpora.get(model_abbr)
    current_bg_for_model = model_bg_counters.get(model_abbr)
    if not current_corpora_for_model or not any(current_corpora_for_model.values()):
        print(f"Warning: Corpora for model '{model_abbr}' is missing or empty. Skipping log odds calculation.")
        continue
    if not current_bg_for_model:
        print(f"Warning: Background counter for model '{model_abbr}' is empty. Skipping log odds calculation.")
        continue
    non_empty_corpora = {method: counter for method, counter in current_corpora_for_model.items() if counter}
    if not non_empty_corpora:
        print(f"Warning: No non-empty method corpora found for model '{model_abbr}' after check. Skipping.")
        continue
    print(f"Calculating log odds for {len(non_empty_corpora)} non-empty methods in {model_abbr}.")
    model_specific_results = logodds(non_empty_corpora, current_bg_for_model)
    if not model_specific_results:
        print(f"Warning: Log odds calculation returned no results for model '{model_abbr}'.")
        calculation_successful = False 
        continue
    for method, scores in model_specific_results.items():
        if scores: all_log_odds_results[f"{model_abbr}_{method}"] = scores
        else: print(f"Note: No log odds scores generated for {model_abbr}_{method}.")
    print(f"--- Finished Log Odds for: {model_abbr} ---")

if not all_log_odds_results and calculation_successful:
     print("\n !!! Warning: Log odds calculation produced no results for any model. Check data. !!!")
elif not all_log_odds_results and not calculation_successful:
    print("\n !!! Error: Log odds calculation produced no results overall and failed. Exiting. !!!")
    if RESPONSES_DIR != './dummy_responses' and not agg_data_loaded : exit()
elif not calculation_successful:
    print("\n !!! Warning: Log odds calculation failed for one or more models. Proceeding with available results. !!!")


# 3. Generate Combined Word Cloud and Bar Plots
if not all_log_odds_results:
    print("No log odds results available to generate combined plots. Skipping visualization.")
elif not individual_processed_responses and any(MODELS_ABBR): 
    print("No individual cleaned response data available for bar plots. Skipping combined visualization.")
else:
    n_models = len(MODELS_ABBR)
    n_methods = len(METHODS)

    if n_models == 0 or n_methods == 0:
        print("No models or methods defined to generate plots for.")
    else:
        rows_per_model = 2 if GENERATE_WORDCLOUDS else 1
        n_plot_rows = n_models * rows_per_model
        
        total_figure_height = 0
        if GENERATE_WORDCLOUDS:
            total_figure_height += (FIGURE_HEIGHT_PER_WC_ROW * n_models)
        total_figure_height += (FIGURE_HEIGHT_PER_BAR_ROW * n_models) 
        
        total_figure_width = FIGURE_WIDTH_PER_COLUMN * n_methods
        
        fig, axes = plt.subplots(nrows=n_plot_rows, ncols=n_methods,
                                 figsize=(total_figure_width, total_figure_height),
                                 squeeze=False)
        print("\nGenerating combined plots...")

        for r_model_idx, model_abbr in enumerate(MODELS_ABBR):
            for c_method_idx, method in enumerate(METHODS):
                corpus_key = f"{model_abbr}_{method}"
                method_log_odds_scores = all_log_odds_results.get(corpus_key, {})
                base_color = COLORBLIND_FRIENDLY_PALETTE.get(method, "#000000")
                
                current_row_offset = r_model_idx * rows_per_model

                if GENERATE_WORDCLOUDS:
                    ax_wc = axes[current_row_offset, c_method_idx]
                    generate_wordcloud_from_scores(method_log_odds_scores, ax_wc, base_color)
                    # --- TITLE REMOVED FROM WORDCLOUD SUBPLOT ---
                    # ax_wc.set_title(f"{model_abbr} - {METHOD_DISPLAY_NAME_MAPPING.get(method, method)}", fontsize=SUBPLOT_TITLE_FONTSIZE, y=1.02)


                bar_chart_row_index = current_row_offset + (1 if GENERATE_WORDCLOUDS else 0)
                ax_bar = axes[bar_chart_row_index, c_method_idx]
                
                sorted_scores = sorted(method_log_odds_scores.items(), key=lambda item: item[1], reverse=True)
                top_n_word_scores_for_bar = dict(sorted_scores[:MAX_WORDS_BAR])
                current_cleaned_response_sets = individual_processed_responses.get(model_abbr, {}).get(method, [])

                if not top_n_word_scores_for_bar:
                    ax_bar.text(0.5, 0.5, "No top words", ha='center', va='center', transform=ax_bar.transAxes, fontsize=BAR_PLOT_YLABEL_FONTSIZE, color='grey')
                    ax_bar.axis("off")
                elif not current_cleaned_response_sets:
                     ax_bar.text(0.5, 0.5, "No cleaned responses", ha='center', va='center', transform=ax_bar.transAxes, fontsize=BAR_PLOT_YLABEL_FONTSIZE, color='grey')
                     ax_bar.axis("off")
                else:
                    generate_response_percentage_barplot_from_cleaned(top_n_word_scores_for_bar, current_cleaned_response_sets, ax_bar, base_color)
                
                # --- TITLE REMOVED FROM BAR CHART SUBPLOT (IF WORDCLOUDS ARE OFF) ---
                # if not GENERATE_WORDCLOUDS:
                    # ax_bar.set_title(f"{model_abbr} - {METHOD_DISPLAY_NAME_MAPPING.get(method, method)}", fontsize=SUBPLOT_TITLE_FONTSIZE, y=1.02)

        # Adjust layout
        fig.subplots_adjust(
            left=0, 
            bottom=0.08 if n_methods > 0 else 0.05, 
            right=1, 
            top=1, # Adjusted top to 1, assuming no figure suptitle
            wspace=0.2, 
            hspace=0.1 if GENERATE_WORDCLOUDS else 0 
        )

        # Add Method Labels at the bottom
        if n_methods > 0:
            for c, method_label_text in enumerate(METHODS):
                ax_in_column = axes[0, c] 
                pos = ax_in_column.get_position() 
                horizontal_center_fig_coords = pos.x0 + pos.width / 2.0
                
                y_pos_method_label = 0.0 
                
                display_name = METHOD_DISPLAY_NAME_MAPPING.get(method_label_text, method_label_text)
                fig.text(horizontal_center_fig_coords, y_pos_method_label, display_name,
                         va='center', ha='center',
                         fontsize=METHOD_LABEL_FONTSIZE)
        
        # --- Save Combined Figure ---
        if not MODELS_ABBR: model_prefix_fig = "unknown_model"
        elif len(MODELS_ABBR) == 1: model_prefix_fig = MODELS_ABBR[0]
        else: model_prefix_fig = "multi_model"
        
        plot_type_suffix = "combined" if GENERATE_WORDCLOUDS else "barcharts_only"
        # Updated filename version
        # Ensure the PLOTS_DIR from config exists
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True) # <--- ADD DIRECTORY CREATION

        output_filename_combined_base = f"{model_prefix_fig}_{MAX_WORDS_CLOUD}wc_{MAX_WORDS_BAR}bar_{plot_type_suffix}_v5"

        output_filename_combined = config.PLOTS_DIR / f"{output_filename_combined_base}.png" # <--- CHANGED
        output_filename_combined_fallback = config.PLOTS_DIR / f"{output_filename_combined_base}_fallback.png" # <--- CHANGED

        try:
            plt.savefig(output_filename_combined, dpi=300, bbox_inches='tight')
            print(f"\nCombined plot generation complete. Saved to {output_filename_combined}")
        except Exception as e:
            print(f"\nError saving combined plot figure: {e}")
            try: 
                plt.savefig(output_filename_combined_fallback, dpi=500)
                print(f"Successfully saved fallback combined plot figure to {output_filename_combined_fallback}")
            except Exception as ef: 
                print(f"Error saving fallback combined plot figure: {ef}")
        plt.close(fig)
        print("Combined plot figure closed.")

print("\n--- Script Execution Finished ---")
