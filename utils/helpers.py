# utils/helpers.py

import os
import json
import re
import math
import string
from collections import Counter
import colorsys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Union

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import argparse

# In utils/helpers.py
import os
import torch
from transformers import set_seed # Ensure this import is in helpers
import pandas as pd

def initialize_environment(cuda_devices_str: str, seed_value: int = 0):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_str
    set_seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value) # Use manual_seed_all for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"INFO: CUDA_VISIBLE_DEVICES set to '{cuda_devices_str}'")
    print(f"INFO: Seeds set to {seed_value}")

def validate_answers(
    example: Dict[str, Any],
    full_responses: List[str],
    labels: List[str],
    answer_phrase: str,
) -> Tuple[int, str, List[str], List[str]]:
    """
    Validates the model's responses against the ground truth using regex for label extraction.
    (Retains the regex logic from the original evaluation scripts)
    """
    ground_answer = example['answer'].lower().replace("(", "").replace(")", "")
    pred_answers_extracted = []
    rationales_extracted = []

    # Compile the regex pattern from labels for efficiency if called many times
    # For a few calls, direct use is fine.
    # Ensure labels are escaped if they can contain regex special characters,
    # though for "real" and "ai-generated" it's not an issue.
    try:
        # Escape each label in case it contains regex metacharacters, then join.
        # However, for simple labels like 'real', 'ai-generated', direct join is fine.
        # If labels could be like "real." then re.escape is important.
        # For current labels, this is fine:
        label_pattern = r"|".join(labels) 
        # If labels could have regex characters:
        # label_pattern = r"|".join(re.escape(lbl) for lbl in labels)
        # For whole word matching, you might add \b:
        # label_pattern = r"\b(" + r"|".join(labels) + r")\b"
    except Exception as e:
        print(f"Error creating regex pattern from labels {labels}: {e}")
        # Fallback or re-raise, depending on desired error handling
        label_pattern = "" # Won't match anything

    for r_text in full_responses:
        rationale = ""
        pred_text_after_phrase = r_text

        if answer_phrase in r_text:
            parts = r_text.split(answer_phrase, 1)
            rationale = parts[0].strip()
            if len(parts) > 1:
                pred_text_after_phrase = parts[1].strip()
        else:
            rationale = ""
            pred_text_after_phrase = r_text.strip()

        extracted_label_match = None
        if label_pattern: # Only search if pattern is valid
            extracted_label_match = re.search(label_pattern, pred_text_after_phrase.lower())
        
        if extracted_label_match:
            pred_answers_extracted.append(extracted_label_match.group(0)) # Use the matched group
        else:
            pred_answers_extracted.append(pred_text_after_phrase) # Fallback

        rationales_extracted.append(rationale)

    if not pred_answers_extracted:
        final_pred_answer = "no_prediction_extracted"
    else:
        final_pred_answer = Counter(pred_answers_extracted).most_common(1)[0][0]

    current_score = 1 if final_pred_answer == ground_answer else 0

    return current_score, final_pred_answer, pred_answers_extracted, rationales_extracted


def update_progress_bar(pbar_instance, current_correct_count: int, current_macro_f1: float):
    """
    Updates a tqdm progress bar with accuracy and macro F1 score.

    Args:
        pbar_instance: The tqdm progress bar instance.
        current_correct_count: Current total number of correct predictions.
        current_macro_f1: Current calculated macro F1 score.
    """
    num_total = pbar_instance.n + 1 # n is number of iterations complete, so +1 for current item
    accuracy = current_correct_count / num_total if num_total > 0 else 0
    pbar_instance.set_description(
        f"Macro-F1: {current_macro_f1:.2f} || " # Original used round(val, 2) which can miss trailing zeros
        f"Accuracy: {accuracy:.2f} ({current_correct_count}/{num_total})"
    )
    pbar_instance.update(1)

def save_evaluation_outputs(
    rationales_data: list,
    score_metrics: dict, # This is the TP, FP, TN, FN dict
    macro_f1_score: float, # The final calculated macro_f1
    model_prefix: str, # "AI_llama" or "AI_qwen"
    dataset_name: str, # e.g., args.dataset
    model_string: str, # e.g., args.llm
    mode_type_str: str, # e.g., args.mode
    num_sequences_val: int, # e.g., args.num
    config_module: any
):
    config_module.RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    config_module.SCORES_DIR.mkdir(parents=True, exist_ok=True)

    base_filename_part = f"{model_prefix}-{dataset_name}-{model_string}-{mode_type_str}-n{num_sequences_val}"

    rationales_filename = f"{base_filename_part}-rationales.jsonl"
    rationales_file_path = config_module.RESPONSES_DIR / rationales_filename
    try:
        with open(rationales_file_path, 'w') as file:
            json.dump(rationales_data, file, indent=4)
        print(f"INFO: Rationales data saved to {rationales_file_path}")
    except IOError as e:
        print(f"ERROR: Could not save rationales to {rationales_file_path}: {e}")


    scores_filename_json = f"{base_filename_part}-scores.json" # For confusion matrix
    scores_file_path_json = config_module.SCORES_DIR / scores_filename_json
    try:
        with open(scores_file_path_json, 'w') as file:
            json.dump(score_metrics, file, indent=4)
        print(f"INFO: Confusion matrix data (JSON) saved to {scores_file_path_json}")
    except IOError as e:
        print(f"ERROR: Could not save confusion matrix to {scores_file_path_json}: {e}")

    # Save the final macro F1 score to a CSV
    # The original scripts saved a dict like {'zeroshot-cot-n1': 0.75}
    # This helper can standardize it.
    csv_scores_dict = {f'{mode_type_str}-n{num_sequences_val}': macro_f1_score}
    scores_df = pd.DataFrame.from_dict(csv_scores_dict, orient='index', columns=['macro_f1'])

    csv_scores_filename = f'{base_filename_part}-scores.csv' # For final macro F1
    csv_file_path = config_module.SCORES_DIR / csv_scores_filename
    try:
        scores_df.to_csv(csv_file_path, index=True, header=True)
        print(f"INFO: Macro F1 score CSV saved to {csv_file_path}")
    except IOError as e:
        print(f"ERROR: Could not save Macro F1 CSV to {csv_file_path}: {e}")

def append_prompt_suffix_for_mode(prompt_text: str, mode_type: str) -> str:
    suffixes = {
        "zeroshot-cot": "Let's think step by step",
        "zeroshot-visualize": "Let's visualize",
        "zeroshot-examine": "Let's examine",
        "zeroshot-pixel": "Let's examine pixel by pixel",
        "zeroshot-zoom": "Let's zoom in",
        "zeroshot-flaws": "Let's examine the flaws",
        "zeroshot-texture": "Let's examine the textures",
        "zeroshot-style": "Let's examine the style",
        "zeroshot-artifacts": "Let's examine the synthesis artifacts",
        "zeroshot-2-artifacts": "Let's examine the style and the synthesis artifacts",
        "zeroshot-3-artifacts": "Let's examine the synthesis artifacts and the style",
        "zeroshot-4-artifacts": "Let's observe the style and the synthesis artifacts",
        "zeroshot-5-artifacts": "Let's inspect the style and the synthesis artifacts",
        "zeroshot-6-artifacts": "Let's survey the style and the synthesis artifacts",
        "zeroshot-7-artifacts": "Let's scrutinize the style and the synthesis artifacts",
        "zeroshot-8-artifacts": "Let's analyze the style and the synthesis artifacts",
        "zeroshot-9-artifacts": "Let's examine the details and the textures",
    }
    # Check for partial matches if mode_type can contain more (e.g. "zeroshot-cot-resize")
    for key_suffix, text_suffix in suffixes.items():
        if key_suffix in mode_type: # Allows mode_type to be like "zeroshot-cot-something"
            prompt_text += f" {text_suffix}" # Add a space before suffix
            break # Add only the first matched suffix if multiple could match.
                 # Or, decide if multiple suffixes can be chained.
    return prompt_text

def load_test_data(dataset_arg: str, config_module: any, question_phrase: str) -> list:
    """Loads test data based on the dataset argument."""
    examples = []
    if 'genimage' in dataset_arg:
        file_path = config_module.GENIMAGE_2K_CSV_FILE if '2k' in dataset_arg else config_module.GENIMAGE_10K_CSV_FILE
        # This function needs to exist in helpers and return the 'examples' list
        examples = helpers.load_genimage_data_examples(file_path, question_phrase)
    elif 'd3' in dataset_arg:
        # This function needs to exist in helpers and return the 'examples' list
        examples = helpers.load_d3_data_examples(config_module.D3_DIR, question_phrase)
    elif 'df40' in dataset_arg:
        file_path = config_module.DF40_2K_CSV_FILE if '2k' in dataset_arg else config_module.DF40_10K_CSV_FILE
        # This function needs to exist in helpers and return the 'examples' list
        examples = helpers.load_df40_data_examples(file_path, question_phrase)
    else:
        print(f"Error: Dataset '{dataset_arg}' not recognized for path configuration in helpers.load_test_data.")
        sys.exit(1)

    random.seed(0) # Ensure consistent shuffling if done here
    random.shuffle(examples)
    print(f"INFO: Loaded and shuffled {len(examples)} examples for dataset '{dataset_arg}'.")
    return examples

def get_evaluation_args_parser():
    parser = argparse.ArgumentParser(description="Vision-Language Model Evaluation Script")
    parser.add_argument("-m", "--mode", type=str, help="Mode of reasoning", default="zeroshot-2-artifacts")
    parser.add_argument("-llm", "--llm", type=str, help="The name of the model", required=True) # Make llm required
    parser.add_argument("-c", "--cuda", type=str, help="CUDA device IDs (e.g., '0' or '0,1')", default="0")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset to use (e.g., 'genimage2k')", required=True) # Make dataset required
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size for model inference", default=10)
    parser.add_argument("-n", "--num", type=int, help="Number of sequences for self-consistency/sampling", default=1)
    # Add any other common arguments here
    return parser

# --- NLTK Resource Management ---
def check_nltk_resource(resource_id: str, download_name: Union[str, None] = None) -> None:
    """
    Checks if an NLTK resource is available and downloads it if not.
    """
    if download_name is None:
        download_name = resource_id.split('/')[-1]
    try:
        if resource_id.startswith('corpora/') or resource_id.startswith('tokenizers/'):
            nltk.data.find(resource_id)
        elif resource_id == 'punkt_tab/english.pickle': # Specific case from distinct_words.py
            nltk.data.find('tokenizers/punkt_tab/english.pickle')
        else:
            nltk.data.find(resource_id) # For others like 'wordnet'
        print(f"NLTK resource '{download_name}' found.")
    except LookupError:
        print(f"NLTK resource '{download_name}' not found. Attempting download...")
        nltk.download(download_name, quiet=True)
        print(f"NLTK resource '{download_name}' downloaded.")
    except Exception as e:
        print(f"An error occurred while checking/downloading NLTK resource {download_name}: {e}")

def ensure_nltk_resources(resources: List[Tuple[str, str]]) -> None:
    """
    Ensures all specified NLTK resources are available.
    Args:
        resources: A list of tuples, where each tuple is (resource_id, download_name).
                   Example: [('corpora/wordnet', 'wordnet'), ('tokenizers/punkt', 'punkt')]
    """
    print("--- Checking NLTK Resources ---")
    for resource_id, download_name in resources:
        check_nltk_resource(resource_id, download_name)
    print("--- NLTK Resource Check Complete ---")

# --- Text Processing ---

def escape_latex(text: Union[str, None]) -> str:
    """
    Escapes characters special to LaTeX.
    Handles $, ^, _, etc.
    """
    if text is None:
        return ''
    text_str = str(text)
    if not text_str:
        return ''
    # Order matters: escape backslash first
    text_str = text_str.replace('\\', r'\textbackslash{}')
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}' # Keep ^ as some scripts might use it for math mode like s$^2$
                                     # If not intended for math, it should be escaped.
                                     # For s$^2$, the input string itself should be r"s$^2$"
    }
    for char, escaped_char in replacements.items():
        text_str = text_str.replace(char, escaped_char)
    return text_str

def remove_non_ascii_characters(text: str) -> str:
    """Removes non-ASCII characters from a string."""
    return re.sub(r'[^\x00-\x7F]+', '', text).replace("  "," ").replace("\n","")

def clean_up_sentence(text: str) -> str:
    """Removes spaces before punctuation marks."""
    return re.sub(r'\s+([,.!?;:])', r'\1', text)

def clean_text(text: str) -> str:
    """Combines non-ASCII removal and punctuation cleanup."""
    text = remove_non_ascii_characters(text)
    text = clean_up_sentence(text)
    return text

def capitalize_first_letter(text: str) -> str:
    """Capitalizes the first letter of a string."""
    if text:
        return text[0].upper() + text[1:]
    return text

def preprocess_text_to_token_set(text: str, lemmatizer_instance: WordNetLemmatizer, manual_skip_words: Set[str] = None) -> Set[str]:
    """
    Preprocesses text and returns a set of unique lemmatized tokens.
    Handles stop words and manual skip words.
    """
    if manual_skip_words is None:
        manual_skip_words = set()

    try:
        stop_words_list = stopwords.words('english')
    except LookupError:
        print("NLTK stopwords not found during preprocess_text_to_token_set. Ensure it's downloaded.")
        # Attempt to download if missing, though ensure_nltk_resources should be called earlier.
        check_nltk_resource('corpora/stopwords', 'stopwords')
        stop_words_list = stopwords.words('english')

    stop_words = set(stop_words_list)
    combined_skip_words = stop_words.union(manual_skip_words)
    processed_token_set = set()

    try:
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
        text = ' '.join(text.split()) # Normalize whitespace
        # Define translator for punctuation removal, including specific quotes
        translator = str.maketrans('', '', string.punctuation + '’‘“”')
        tokens = word_tokenize(text.lower())

        for token in tokens:
            token = token.translate(translator) # Remove punctuation
            if token.isalpha() and token not in combined_skip_words: # Check if alphabetic and not a stop/skip word
                try:
                    lemma = lemmatizer_instance.lemmatize(token)
                    if lemma and len(lemma) > 1 and lemma not in combined_skip_words: # Ensure lemma is valid
                        processed_token_set.add(lemma)
                except Exception as lem_e:
                    print(f"Error lemmatizing token '{token}': {lem_e}")
    except Exception as e:
        print(f"Error in preprocess_text_to_token_set for text starting with: '{str(text)[:50]}...' - Error: {e}")
        return set()
    return processed_token_set

# --- Score Calculation & Formatting ---

def _extract_answers_for_f1(rationales_data_sample: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Extracts predicted and ground truth answers from rationale data.
    Internal helper for F1 calculation functions.
    """
    if not rationales_data_sample:
        return [], []
    pred_answers, ground_answers = [], []
    for item in rationales_data_sample:
        pred = item.get('pred_answer')
        ground = item.get('ground_answer')
        if pred is not None and ground is not None and isinstance(pred, str) and isinstance(ground, str):
            pred_answers.append(pred.lower())
            ground_answers.append(ground.lower())
    return pred_answers, ground_answers

def calculate_macro_f1_score_from_answers(pred_answers: List[str], ground_answers: List[str], possible_labels: List[str] = None) -> Union[float, None]:
    """
    Calculates macro F1 score from lists of predicted and ground truth answers.
    Args:
        pred_answers: List of predicted answer strings.
        ground_answers: List of ground truth answer strings.
        possible_labels: List of possible labels for F1 score calculation. Defaults to ['real', 'ai-generated'].
    Returns:
        Macro F1 score (0-100) or None if calculation is not possible.
    """
    if not pred_answers or not ground_answers or len(pred_answers) != len(ground_answers):
        return None
    if possible_labels is None:
        possible_labels = ['real', 'ai-generated']
    try:
        # zero_division=0 returns 0.0 if all predictions and labels are negative for a class.
        # zero_division=1 returns 1.0 in that case.
        # Consider what behavior is desired. 0 seems more common for "no positive examples".
        score = f1_score(ground_answers, pred_answers, labels=possible_labels, average='macro', zero_division=0)
        return round(score * 100, 1) # Return as percentage rounded to 1 decimal place
    except Exception as e:
        print(f"    Error calculating F1 score: {e}", file=sys.stderr)
        return None

def calculate_f1_from_rationales(rationales_list: List[Dict[str, Any]], possible_labels: List[str] = None) -> Tuple[Union[float, None], int]:
    """
    Calculates F1 score from a list of rationale objects.
    Each rationale object should be a dict with 'pred_answer' and 'ground_answer'.
    Args:
        rationales_list: List of rationale dictionaries.
        possible_labels: List of possible labels for F1 score calculation.
    Returns:
        A tuple of (F1 score (0-100) or None, number of samples).
    """
    if not rationales_list:
        return None, 0
    pred_answers, ground_answers = _extract_answers_for_f1(rationales_list)
    n_samples = len(ground_answers)
    if n_samples == 0:
        return None, 0
    f1 = calculate_macro_f1_score_from_answers(pred_answers, ground_answers, possible_labels)
    return f1, n_samples


def format_score_for_display(score: Union[float, int, None], zero_pad: bool = True, decimal_places: int = 1) -> str:
    """
    Formats a numerical score for display in tables or plots.
    Handles NaN/None, zero padding for single digits, and rounding.
    Args:
        score: The score to format.
        zero_pad: If True, pads single-digit scores with a leading zero (e.g., 7.0 -> 07.0).
        decimal_places: Number of decimal places to round to.
    Returns:
        A string representation of the formatted score, or '-'.
    """
    if pd.isna(score) or score is None:
        return '-'
    try:
        score_float = float(score)
    except ValueError:
        return '-' # Should not happen if input is float/int/None

    formatted_val = f"{score_float:.{decimal_places}f}"

    if zero_pad and score_float >= 0 and score_float < 10:
        # Check to avoid double padding like "007.0" if already "07.0" due to formatting
        if not formatted_val.startswith('0') or (decimal_places > 0 and formatted_val.startswith('0.')):
             # Check if it's an integer like 7.0, 8.0 before padding
            if f'.{"0"*decimal_places}' in formatted_val: # e.g. "7.0", "8.00"
                 formatted_val = f"0{formatted_val}"
            # This part handles cases where score_float might be like 7 (no decimal part after initial formatting)
            elif '.' not in formatted_val and decimal_places > 0:
                 formatted_val = f"0{score_float:.{decimal_places}f}" # Reformat to ensure .0
            elif '.' not in formatted_val and decimal_places == 0: # Integer, e.g. 7
                formatted_val = f"0{formatted_val}"


    return formatted_val

def get_macro_f1_from_counts(score_counts: Dict[str, int]) -> float:
    """
    Calculates the macro F1 score from a dictionary of TP, FP, TN, FN counts.
    Used by evaluation scripts.
    Args:
        score_counts: A dictionary with keys 'TP', 'FP', 'TN', 'FN' and their integer counts.
    Returns:
        Macro F1 score (0-1).
    """
    tp = score_counts.get('TP', 0)
    fp = score_counts.get('FP', 0)
    tn = score_counts.get('TN', 0)
    fn = score_counts.get('FN', 0)

    # F1 for positive class (real)
    prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    reca_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_pos = 2 * prec_pos * reca_pos / (prec_pos + reca_pos) if (prec_pos + reca_pos) > 0 else 0

    # F1 for negative class (ai-generated)
    # For the negative class, effectively: TN becomes "TP_neg", FN becomes "FP_neg", FP becomes "FN_neg"
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0 # Correctly predicted AI / All predicted as AI
    reca_neg = tn / (tn + fp) if (tn + fp) > 0 else 0 # Correctly predicted AI / All actual AI
    f1_neg = 2 * prec_neg * reca_neg / (prec_neg + reca_neg) if (prec_neg + reca_neg) > 0 else 0

    macro_f1 = (f1_pos + f1_neg) / 2.0
    return macro_f1


# --- Data Loading ---

def load_dataset_csv_mapping(file_path: Union[str, Path], image_col: str, subset_col: str, label_col: Union[str, None] = None, real_label_value: Union[str, None] = None) -> Dict[str, str]:
    """
    Generic function to load dataset CSV and create an image_path to subset_name mapping.
    Handles cases where 'real' images have their subset in 'label_col' or a more general 'subset_col'.
    Args:
        file_path: Path to the CSV file.
        image_col: Name of the column containing the image identifier (path or name).
        subset_col: Name of the column containing the subset/generator name for AI images.
        label_col: (Optional) Name of the column containing the 'real'/'ai-generated' label.
        real_label_value: (Optional) The value in 'label_col' that identifies real images (e.g., 'real').
    Returns:
        A dictionary mapping image identifiers to lowercase subset names.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Data file not found: {file_path}", file=sys.stderr)
        return {}
    try:
        data = pd.read_csv(file_path)
        subset_data = {}
        required_cols = [image_col, subset_col]
        if label_col:
            required_cols.append(label_col)

        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            print(f"Error: CSV {file_path} missing required columns: {missing}", file=sys.stderr)
            return {}

        for _, row in data.iterrows():
            image_identifier = row[image_col]
            raw_subset = 'unknown'

            is_real = False
            if label_col and real_label_value and pd.notna(row[label_col]):
                if str(row[label_col]).lower() == real_label_value.lower():
                    is_real = True
            
            if is_real:
                raw_subset = real_label_value # or just 'real'
            elif pd.notna(row[subset_col]):
                raw_subset = row[subset_col]
            else:
                print(f"Warning: Missing subset/generator value for image {image_identifier} in {file_path} and not identified as real.", file=sys.stderr)
            
            subset_data[image_identifier] = str(raw_subset).lower()
        return subset_data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}", file=sys.stderr)
        return {}

def load_genimage_data_mapping(file_path: Union[str, Path]) -> Dict[str, str]:
    """Loads GenImage data mapping image paths to lowercase subset names."""
    return load_dataset_csv_mapping(file_path, image_col='img_path', subset_col='dataset', label_col='dataset', real_label_value='real')

def load_df40_data_mapping(file_path: Union[str, Path]) -> Dict[str, str]:
    """Loads DF40 data mapping image paths to lowercase subset names."""
    return load_dataset_csv_mapping(file_path, image_col='file_path', subset_col='dataset', label_col='label', real_label_value='real')


def load_d3_data_mapping(directory_path: Union[str, Path]) -> Dict[str, str]:
    """
    Loads D3 data mapping image filenames (not full paths) to lowercase subset names.
    Subsets are inferred from filenames like 'fake_biggan_0.png' -> 'biggan', 'real_ILSVRC...png' -> 'real'.
    """
    print(f"\n--- Loading D3 image-to-subset mapping from directory: {directory_path} ---")
    if not os.path.isdir(directory_path):
        print(f"Warning: D3 data directory not found: {directory_path}", file=sys.stderr)
        return {}

    subset_data = {}
    processed_png_count = 0
    try:
        for filename in os.listdir(directory_path):
            if filename.endswith(".png"):
                processed_png_count += 1
                base_name = filename.replace('.png', '')
                parts = base_name.split('_')
                raw_subset = 'unknown_d3_subset'

                # D3 Naming convention interpretation (example based)
                # e.g., real_ILSVRC2012_val_00000001.png -> real
                #       fake_biggan_0.png -> biggan
                #       fake_stable_diffusion_v_1_4_100.png -> stable_diffusion_v_1_4
                if 'real' in parts[0].lower() and len(parts) > 1: # Starts with 'real_'
                    raw_subset = 'real'
                elif 'fake' in parts[0].lower() and len(parts) > 1: # Starts with 'fake_'
                    # Attempt to join parts after "fake" and before potential numeric suffix
                    potential_subset_parts = []
                    for part in parts[1:]:
                        if part.isdigit() or (part.startswith('gen') and part[3:].isdigit()): # Heuristic for trailing numbers or "gen<number>"
                            break
                        potential_subset_parts.append(part)
                    
                    if potential_subset_parts:
                        raw_subset = "_".join(potential_subset_parts)
                    elif len(parts) > 1 : # Fallback if only 'fake_something'
                        raw_subset = parts[1]

                else: # Unrecognized pattern
                    print(f"Warning: D3 file '{filename}' has an unrecognized naming pattern. Parts: {parts}. Assigning to '{raw_subset}'.", file=sys.stderr)

                subset_data[filename] = str(raw_subset).lower() # Key by filename

    except Exception as e:
        print(f"Error listing or processing D3 directory {directory_path}: {e}", file=sys.stderr)
        return {}

    print(f"Processed {processed_png_count} PNG files for D3 mapping. Found {len(subset_data)} mappings.")
    # print(f"Unique D3 subsets extracted by load_d3_data_mapping: {sorted(list(set(subset_data.values())))}")
    return subset_data


def load_scores_from_jsonl_file(file_path: Union[str, Path]) -> Dict[str, int]:
    """
    Loads image scores from a JSONL file or a JSON file containing a list of records.
    Each record is expected to be a JSON object with "image" (path) and "cur_score" (integer).
    Args:
        file_path: Path to the JSON or JSONL file.
    Returns:
        A dictionary mapping image paths to their cur_score. Empty if errors.
    """
    scores = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        all_records = []
        try: # Try parsing as a single JSON array first
            data = json.loads(content)
            if isinstance(data, list):
                all_records = data
            elif isinstance(data, dict): # Handle case where file might be a single dict with a 'rationales' key
                if 'rationales' in data and isinstance(data['rationales'], list):
                    all_records = data['rationales']
                else: # Or a single record itself
                    all_records = [data]
            else:
                print(f"Warning: Content of {file_path} is not a JSON list or expected dict structure. Trying JSONL.", file=sys.stderr)
                # Fall through to JSONL parsing
        except json.JSONDecodeError:
            # If fails, assume JSONL and parse line by line
            print(f"Info: Could not parse {file_path} as a single JSON document. Assuming JSONL format.", file=sys.stderr)
            pass # Fall through to JSONL parsing

        if not all_records: # If still no records (either due to JSON parse fail or it was truly JSONL)
            for line_num, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    all_records.append(record)
                except json.JSONDecodeError as e_line:
                    print(f"Warning: Skipping line {line_num} in {file_path} due to JSON decode error: {e_line}. Line: '{line[:100]}...'", file=sys.stderr)
                    continue
        
        if not all_records:
            print(f"Warning: No records found or parsed from {file_path}.", file=sys.stderr)
            return scores

        for record_num, record in enumerate(all_records, 1):
            if not isinstance(record, dict):
                print(f"Warning: Record {record_num} in {file_path} is not a dict. Skipping.", file=sys.stderr)
                continue
            
            image_path = record.get("image")
            cur_score_val = record.get("cur_score")

            if image_path is None or cur_score_val is None:
                # print(f"Warning: Record {record_num} in {file_path} missing 'image' or 'cur_score'. Skipping.", file=sys.stderr)
                continue
            if not isinstance(image_path, str):
                # print(f"Warning: Record {record_num} 'image' path is not a string. Skipping.", file=sys.stderr)
                continue
            
            try:
                cur_score = int(cur_score_val)
                scores[image_path] = cur_score
            except (ValueError, TypeError):
                # print(f"Warning: Record {record_num} 'cur_score' is not a valid integer ('{cur_score_val}'). Skipping.", file=sys.stderr)
                continue
                
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}", file=sys.stderr)
    
    return scores


# --- Plotting & Color Utilities ---

def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Converts a hex color string to an RGB tuple (values 0-1)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3: # Expand shorthand hex (e.g., #03F to #0033FF)
        hex_color = ''.join([c*2 for c in hex_color])
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color format: {hex_color}")
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb_color: Tuple[float, float, float]) -> str:
    """Converts an RGB tuple (values 0-1) to a hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb_color[0]*255), int(rgb_color[1]*255), int(rgb_color[2]*255)
    )

def adjust_lightness(rgb_color: Tuple[float, float, float], factor: float) -> Tuple[float, float, float]:
    """
    Adjusts the lightness of an RGB color.
    Factor 0.0 is darkest (but not black), 1.0 is lightest (but not white).
    Lightness is scaled between min_lightness and max_lightness.
    """
    try:
        h, l, s = colorsys.rgb_to_hls(*rgb_color)
        # Define the range of lightness to map to.
        # Prevents colors from becoming pure black or pure white, maintaining some hue.
        min_target_lightness = 0.15 # Darkest version will be this light
        max_target_lightness = 0.85 # Lightest version will be this light
        
        # Interpolate lightness: factor=0 -> max_target_lightness, factor=1 -> min_target_lightness
        # This means higher score (if factor is score-based and normalized 0-1) -> darker color
        # If you want higher score -> lighter color, it should be:
        # new_l = min_target_lightness + factor * (max_target_lightness - min_target_lightness)
        new_l = max_target_lightness - factor * (max_target_lightness - min_target_lightness)
        
        new_l = max(0.05, min(0.95, new_l)) # Clamp to avoid extremes if input factor is outside 0-1
        return colorsys.hls_to_rgb(h, new_l, s)
    except Exception as e:
        print(f"Error adjusting lightness for color {rgb_color} with factor {factor}: {e}")
        return rgb_color # Return original color on error

def wordcloud_color_func_factory(base_color_hex: str, word_scores: Dict[str, float]):
    """
    Factory for creating a color function for WordCloud based on word scores.
    Words with higher scores (more distinctive) will be rendered darker.
    """
    try:
        base_color_rgb = hex_to_rgb(base_color_hex)
    except ValueError as e:
        print(f"Error converting base hex color {base_color_hex} for wordcloud: {e}. Using black as fallback.")
        base_color_rgb = (0.0, 0.0, 0.0) # Fallback to black

    # Consider only positive scores for normalization, as negative scores might not be meaningful for "distinctiveness"
    positive_scores = {word: score for word, score in word_scores.items() if score > 0}

    if not positive_scores: # No positive scores, all words will use the darkest shade (factor = 1.0)
        min_score, max_score, score_range = 0, 0, 1.0 # Avoid division by zero
    else:
        scores_values = list(positive_scores.values())
        min_score = min(scores_values)
        max_score = max(scores_values)
        score_range = max_score - min_score
        if score_range <= 1e-6: # Handle case where all positive scores are (nearly) identical
            score_range = 1.0 # Avoid division by zero, effectively makes all words with positive score use same intermediate shade

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        score = word_scores.get(word, 0)
        
        if score <= 0 or not positive_scores: # Word has no score or no positive scores exist in the set
            lightness_factor = 1.0 # Use the darkest shade (max_target_lightness - 1.0 * range)
        else:
            # Normalize the score within the range of positive scores
            # A higher score should result in a darker color (closer to min_target_lightness)
            # So, a higher score means factor closer to 1.0
            clamped_score = max(min_score, min(score, max_score)) # Clamp score to be within the observed positive range
            lightness_factor = (clamped_score - min_score) / score_range if score_range > 0 else 0.5 # Normalize (0 to 1)

        adjusted_rgb = adjust_lightness(base_color_rgb, lightness_factor)
        return rgb_to_hex(adjusted_rgb)
    return color_func


# --- Model Name Parsing ---
def parse_model_name(model_name_config: str) -> Tuple[str, str]:
    """
    Parses a model configuration string into family and size.
    Example: "qwen25-7b" -> ("qwen2.5", "7b")
             "llama3-11b" -> ("llama3", "11b") (adjust if llama3.2 is standard)
    """
    match = re.match(r'([a-zA-Z]+)(\d+(\.\d+)?)?-?(\d+[bBmB])', model_name_config)
    if match:
        base_name, version_with_dot, _, size_str = match.groups()
        version = version_with_dot if version_with_dot else "" # e.g., "2.5" or ""

        # Specific family naming conventions
        if base_name.lower() == 'qwen' and version == '25': # from prompt_table and model_size
            family_name = 'qwen2.5'
        elif base_name.lower() == 'qwen' and version == '2.5': # from prompt_table (alternative)
             family_name = 'qwen2.5'
        elif base_name.lower() == 'llama' and version == '3': # from prompt_table (llama3-11b)
            # Decide if "llama3" or "llama3.2" is the canonical family name
            family_name = 'llama3' # or 'llama3.2' if that's preferred
        elif base_name.lower() == 'llama' and (not version or version == '3.2'): # from model_size_table (llama3.2)
            family_name = 'llama3.2'
        elif version:
            family_name = f"{base_name.lower()}{version}"
        else:
            family_name = base_name.lower()
        
        return family_name, size_str.lower()
    
    # Fallback for names like "CoDE" or unparseable
    if model_name_config.upper() == "CODE":
        return "code", "6m" # Example size, adjust as needed for CoDE

    print(f"Warning: Could not parse family/size reliably from LLM name: {model_name_config}. Returning as is.", file=sys.stderr)
    return model_name_config.lower(), "unknown_size"


if __name__ == '__main__':
    # Example usage of some utility functions
    print("--- Utility Function Examples ---")

    # NLTK resources
    # ensure_nltk_resources([('corpora/wordnet', 'wordnet'), ('tokenizers/punkt', 'punkt'), ('corpora/stopwords', 'stopwords')])

    # Text processing
    latex_example = "This is a _test_ with $maths^2$ & special chars."
    print(f"Original: '{latex_example}' -> LaTeX escaped: '{escape_latex(latex_example)}'")
    
    text_to_clean = "  This sentence needs cleaning!!   Right?  "
    print(f"Original: '{text_to_clean}' -> Cleaned: '{clean_text(text_to_clean)}'")

    # Score formatting
    print(f"Score 7.0 formatted: {format_score_for_display(7.0)}")
    print(f"Score 7 formatted: {format_score_for_display(7)}")
    print(f"Score 12.345 formatted (1dp): {format_score_for_display(12.345, decimal_places=1)}")
    print(f"Score 12.355 formatted (2dp, no pad): {format_score_for_display(12.355, zero_pad=False, decimal_places=2)}")
    print(f"Score None formatted: {format_score_for_display(None)}")

    # F1 calculation
    preds = ["real", "ai-generated", "real", "real"]
    grounds = ["real", "real", "ai-generated", "real"]
    f1 = calculate_macro_f1_score_from_answers(preds, grounds)
    print(f"F1 score for {preds} vs {grounds}: {f1}")

    rationales_sample = [
        {'pred_answer': 'real', 'ground_answer': 'real'},
        {'pred_answer': 'ai-generated', 'ground_answer': 'real'},
        {'pred_answer': 'real', 'ground_answer': 'ai-generated'},
        {'pred_answer': 'real', 'ground_answer': 'real'}
    ]
    f1_rat, n_rat = calculate_f1_from_rationales(rationales_sample)
    print(f"F1 from rationales: {f1_rat}, N_samples: {n_rat}")
    
    # Model name parsing
    print(f"Parse 'qwen25-7b': {parse_model_name('qwen25-7b')}")
    print(f"Parse 'llama3-11b': {parse_model_name('llama3-11b')}")
    print(f"Parse 'CoDE': {parse_model_name('CoDE')}")

    print("--- End of Examples ---")