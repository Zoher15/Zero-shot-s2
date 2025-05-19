# utils/helpers.py

import os
import json
import re
import math
import string
from collections import Counter, defaultdict # defaultdict was missing from imports
import colorsys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Union

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import nltk # Keep nltk import at top level
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import argparse
import torch
from transformers import set_seed
import logging # Import logging

# --- Standard Logger for this Module ---
logger = logging.getLogger(__name__)
# Basic configuration for the logger if not configured by the calling script
# This ensures helpers can log even if the main script doesn't set up logging.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

def initialize_environment(cuda_devices_str: str, seed_value: int = 0):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_str
    set_seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"CUDA_VISIBLE_DEVICES set to '{cuda_devices_str}'")
    logger.info(f"Seeds set to {seed_value}")

def validate_answers(
    example: Dict[str, Any],
    full_responses: List[str],
    labels: List[str],
    answer_phrase: str
) -> Tuple[int, str, List[str], List[str]]:
    ground_answer = example['answer'].lower().replace("(", "").replace(")", "")
    pred_answer_final = None
    pred_answers_extracted = []
    rationales_extracted = []

    for r_text in full_responses:
        rationale = ""
        pred_text_after_phrase = r_text

        if answer_phrase in r_text:
            parts = r_text.split(answer_phrase, 1)
            rationale = parts[0].strip()
            if len(parts) > 1:
                pred_text_after_phrase = parts[1].strip()
        else:
            pred_text_after_phrase = r_text.strip()

        regex_pattern = r"|".join(map(re.escape, labels)) # Escape labels for regex
        match_obj = re.search(regex_pattern, pred_text_after_phrase.lower())

        extracted_label_for_current_response = pred_text_after_phrase # Default if no specific label found
        if match_obj:
            extracted_label_for_current_response = match_obj.group(0)

        pred_answers_extracted.append(extracted_label_for_current_response)
        rationales_extracted.append(rationale)

    if not pred_answers_extracted:
        logger.warning(f"No answers extracted from responses for example: {example.get('image', 'Unknown image')}")
        final_pred_answer = "no_valid_prediction" # Or some other default
    else:
        count = Counter(pred_answers_extracted)
        if not count:
            logger.warning(f"Answer counter empty for example: {example.get('image', 'Unknown image')}, extracted: {pred_answers_extracted}")
            final_pred_answer = "counter_empty_fallback"
        else:
            final_pred_answer = count.most_common(1)[0][0]

    current_score = 1 if final_pred_answer == ground_answer else 0
    return current_score, final_pred_answer, pred_answers_extracted, rationales_extracted


def update_progress(pbar, correct_count: int, macro_f1_percentage: float):
    ntotal = pbar.n + 1 # pbar.n is the number of iterations completed
    # Ensure macro_f1_percentage is a float for formatting
    description = f"Macro-F1: {float(macro_f1_percentage):.2f}% || Accuracy: {correct_count/ntotal:.2%} ({correct_count}/{ntotal})"
    pbar.set_description(description)
    pbar.update()


def save_evaluation_outputs(
    rationales_data: list,
    score_metrics: dict,
    macro_f1_score: float,
    model_prefix: str,
    dataset_name: str,
    model_string: str,
    mode_type_str: str,
    num_sequences_val: int,
    config_module: Any # Should be the imported config module
):
    # Ensure directories exist
    try:
        config_module.RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
        config_module.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create output directories: {e}", exc_info=True)
        return # Cannot proceed if directories can't be made

    base_filename_part = f"{model_prefix}-{dataset_name}-{model_string}-{mode_type_str}-n{num_sequences_val}"

    rationales_filename = f"{base_filename_part}-rationales.jsonl"
    rationales_file_path = config_module.RESPONSES_DIR / rationales_filename
    try:
        with open(rationales_file_path, 'w', encoding='utf-8') as file: # Added encoding
            json.dump(rationales_data, file, indent=4)
        logger.info(f"Rationales data saved to {rationales_file_path}")
    except IOError as e:
        logger.error(f"Could not save rationales to {rationales_file_path}: {e}", exc_info=True)

    scores_filename_json = f"{base_filename_part}-scores.json"
    scores_file_path_json = config_module.SCORES_DIR / scores_filename_json
    try:
        with open(scores_file_path_json, 'w', encoding='utf-8') as file: # Added encoding
            json.dump(score_metrics, file, indent=4)
        logger.info(f"Confusion matrix data (JSON) saved to {scores_file_path_json}")
    except IOError as e:
        logger.error(f"Could not save confusion matrix to {scores_file_path_json}: {e}", exc_info=True)

    csv_scores_dict = {f'{mode_type_str}-n{num_sequences_val}': macro_f1_score}
    scores_df = pd.DataFrame.from_dict(csv_scores_dict, orient='index', columns=['macro_f1'])

    csv_scores_filename = f'{base_filename_part}-scores.csv'
    csv_file_path = config_module.SCORES_DIR / csv_scores_filename
    try:
        scores_df.to_csv(csv_file_path, index=True, header=True)
        logger.info(f"Macro F1 score CSV saved to {csv_file_path}")
    except IOError as e:
        logger.error(f"Could not save Macro F1 CSV to {csv_file_path}: {e}", exc_info=True)

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
    suffix_to_add = ""
    for key_suffix, text_suffix in suffixes.items():
        if key_suffix in mode_type:
            suffix_to_add = text_suffix
            break
    if suffix_to_add: # Ensure space only if suffix is added
        return f"{prompt_text.rstrip()} {suffix_to_add}" # rstrip to avoid double spaces if prompt_text ends with one
    return prompt_text


def load_genimage_data_examples(file_path: Union[str, Path], question_str: str) -> List[Dict[str, Any]]:
    data = pd.read_csv(Path(file_path))
    examples = []
    for _, row in data.iterrows():
        example_data = {}
        example_data['image'] = str(row['img_path'])
        example_data['question'] = question_str
        example_data['answer'] = 'real' if str(row['dataset']).lower() == 'real' else 'ai-generated'
        examples.append(example_data)
    return examples

def load_d3_data_examples(dir_path: Union[str, Path], question_str: str) -> List[Dict[str, Any]]:
    examples = []
    directory = Path(dir_path)
    if not directory.is_dir():
        logger.error(f"D3 data directory not found: {directory}")
        return examples

    for file_name in os.listdir(directory):
        if file_name.lower().endswith(".png"):
            full_path = directory / file_name
            answer = 'real' if 'real' in file_name.lower() else 'ai-generated'
            examples.append({
                'image': str(full_path),
                'question': question_str,
                'answer': answer
            })
    return examples

def load_df40_data_examples(file_path: Union[str, Path], question_str: str) -> List[Dict[str, Any]]:
    data = pd.read_csv(Path(file_path))
    examples = []
    for _, row in data.iterrows():
        example_data = {}
        example_data['image'] = str(row['file_path'])
        example_data['question'] = question_str
        example_data['answer'] = 'real' if str(row['label']).lower() == 'real' else 'ai-generated'
        examples.append(example_data)
    return examples

def load_faces_data_examples(data_dir_path: Union[str, Path], question_str: str) -> List[Dict[str, Any]]:
    """Loads FACES dataset examples from specified subdirectories."""
    data_dir = Path(data_dir_path)
    dirs_info = [("LD_raw_512Size", "ai-generated"),
                 ("StyleGAN_raw_512size", "ai-generated"),
                 ("Real_512Size", "real")]
    examples = []
    for dir_name, answer_label in dirs_info:
        current_subdir = data_dir / dir_name
        if not current_subdir.is_dir():
            logger.warning(f"FACES subdirectory not found: {current_subdir}")
            continue
        for f_name in os.listdir(current_subdir):
            # Consider adding checks for file extensions if needed (e.g., .png, .jpg)
            examples.append({
                'image': str(current_subdir / f_name.strip()),
                'question': question_str,
                'answer': answer_label
            })
    return examples


def load_test_data(dataset_arg: str, config_module: Any, question_phrase: str) -> list:
    examples = []
    if 'genimage' in dataset_arg:
        file_path = config_module.GENIMAGE_2K_CSV_FILE if '2k' in dataset_arg else config_module.GENIMAGE_10K_CSV_FILE
        examples = load_genimage_data_examples(file_path, question_phrase)
    elif 'd3' in dataset_arg:
        examples = load_d3_data_examples(config_module.D3_DIR, question_phrase)
    elif 'df40' in dataset_arg:
        file_path = config_module.DF40_2K_CSV_FILE if '2k' in dataset_arg else config_module.DF40_10K_CSV_FILE
        examples = load_df40_data_examples(file_path, question_phrase)
    elif 'faces' in dataset_arg: # Added FACES dataset loading
        faces_dir = config_module.DATA_DIR / "FACES" # Assuming FACES is directly under DATA_DIR
        examples = load_faces_data_examples(faces_dir, question_phrase)
    else:
        logger.error(f"Dataset '{dataset_arg}' not recognized for path configuration in helpers.load_test_data.")
        sys.exit(1)

    random.seed(0)
    random.shuffle(examples)
    logger.info(f"Loaded and shuffled {len(examples)} examples for dataset '{dataset_arg}'.")
    return examples

def get_evaluation_args_parser() -> argparse.ArgumentParser: # Added return type hint
    parser = argparse.ArgumentParser(description="Vision-Language Model Evaluation Script")
    parser.add_argument("-m", "--mode", type=str, help="Mode of reasoning", default="zeroshot-2-artifacts")
    parser.add_argument("-llm", "--llm", type=str, help="The name of the model", required=True)
    parser.add_argument("-c", "--cuda", type=str, help="CUDA device IDs (e.g., '0' or '0,1')", default="0")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset to use (e.g., 'genimage2k')", required=True)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size for model inference", default=10)
    parser.add_argument("-n", "--num", type=int, help="Number of sequences for self-consistency/sampling", default=1)
    return parser

def check_nltk_resource(resource_id: str, download_name: Union[str, None] = None) -> None:
    if download_name is None:
        download_name = resource_id.split('/')[-1]
    try:
        if resource_id.startswith('corpora/') or resource_id.startswith('tokenizers/'):
            nltk.data.find(resource_id)
        elif resource_id == 'punkt_tab/english.pickle':
            nltk.data.find('tokenizers/punkt_tab/english.pickle')
        else:
            nltk.data.find(resource_id)
        logger.info(f"NLTK resource '{download_name}' found.")
    except LookupError:
        logger.info(f"NLTK resource '{download_name}' not found. Attempting download...")
        nltk.download(download_name, quiet=True)
        logger.info(f"NLTK resource '{download_name}' downloaded.")
    except Exception as e:
        logger.error(f"An error occurred while checking/downloading NLTK resource {download_name}: {e}", exc_info=True)

def ensure_nltk_resources(resources: List[Tuple[str, str]]) -> None:
    logger.info("--- Checking NLTK Resources ---")
    for resource_id, download_name in resources:
        check_nltk_resource(resource_id, download_name)
    logger.info("--- NLTK Resource Check Complete ---")

# --- Text Processing ---
def escape_latex(text: Union[str, None]) -> str:
    if text is None: return ''
    text_str = str(text)
    if not text_str: return ''
    text_str = text_str.replace('\\', r'\textbackslash{}')
    replacements = { '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    for char, escaped_char in replacements.items():
        text_str = text_str.replace(char, escaped_char)
    return text_str

# ... (other text processing functions like remove_non_ascii_characters, clean_text, capitalize_first_letter) ...
# They can also use logger.warning or logger.debug if they encounter issues or for verbose output.

def preprocess_text_to_token_set(text: str, lemmatizer_instance: WordNetLemmatizer, manual_skip_words: Set[str] = None) -> Set[str]:
    if manual_skip_words is None: manual_skip_words = set()
    try:
        stop_words_list = stopwords.words('english')
    except LookupError:
        logger.error("NLTK stopwords not found during preprocess_text_to_token_set. Ensure it's downloaded.")
        check_nltk_resource('corpora/stopwords', 'stopwords') # Attempt download
        stop_words_list = stopwords.words('english')

    stop_words = set(stop_words_list)
    combined_skip_words = stop_words.union(manual_skip_words)
    processed_token_set = set()

    try:
        if not isinstance(text, str): text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = ' '.join(text.split())
        translator = str.maketrans('', '', string.punctuation + '’‘“”')
        tokens = word_tokenize(text.lower())

        for token in tokens:
            token = token.translate(translator)
            if token.isalpha() and token not in combined_skip_words:
                try:
                    lemma = lemmatizer_instance.lemmatize(token)
                    if lemma and len(lemma) > 1 and lemma not in combined_skip_words:
                        processed_token_set.add(lemma)
                except Exception as lem_e:
                    logger.error(f"Error lemmatizing token '{token}': {lem_e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error in preprocess_text_to_token_set for text starting with: '{str(text)[:50]}...' - Error: {e}", exc_info=True)
        return set()
    return processed_token_set

# --- Score Calculation & Formatting ---
def _extract_answers_for_f1(rationales_data_sample: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    if not rationales_data_sample: return [], []
    pred_answers, ground_answers = [], []
    for item in rationales_data_sample:
        pred = item.get('pred_answer')
        ground = item.get('ground_answer')
        if pred is not None and ground is not None and isinstance(pred, str) and isinstance(ground, str):
            pred_answers.append(pred.lower())
            ground_answers.append(ground.lower())
    return pred_answers, ground_answers

def calculate_macro_f1_score_from_answers(pred_answers: List[str], ground_answers: List[str], possible_labels: List[str] = None) -> Union[float, None]:
    if not pred_answers or not ground_answers or len(pred_answers) != len(ground_answers):
        return None
    if possible_labels is None: possible_labels = ['real', 'ai-generated']
    try:
        score = f1_score(ground_answers, pred_answers, labels=possible_labels, average='macro', zero_division=0)
        return round(score * 100, 1)
    except Exception as e:
        logger.error(f"Error calculating F1 score: {e}", exc_info=True)
        return None

def calculate_f1_from_rationales(rationales_list: List[Dict[str, Any]], possible_labels: List[str] = None) -> Tuple[Union[float, None], int]:
    if not rationales_list: return None, 0
    pred_answers, ground_answers = _extract_answers_for_f1(rationales_list)
    n_samples = len(ground_answers)
    if n_samples == 0: return None, 0
    f1 = calculate_macro_f1_score_from_answers(pred_answers, ground_answers, possible_labels)
    return f1, n_samples

def format_score_for_display(score: Union[float, int, None], zero_pad: bool = True, decimal_places: int = 1) -> str:
    # This function seems fine as is.
    if pd.isna(score) or score is None: return '-'
    try:
        score_float = float(score)
    except ValueError: return '-'
    formatted_val = f"{score_float:.{decimal_places}f}"
    if zero_pad and 0 <= score_float < 10:
        if not formatted_val.startswith('0') or (decimal_places > 0 and formatted_val.startswith('0.')):
            if f'.{"0"*decimal_places}' in formatted_val:
                 formatted_val = f"0{formatted_val}"
            elif '.' not in formatted_val and decimal_places > 0:
                 formatted_val = f"0{score_float:.{decimal_places}f}"
            elif '.' not in formatted_val and decimal_places == 0:
                formatted_val = f"0{formatted_val}"
    return formatted_val

def get_macro_f1_from_counts(score_counts: Dict[str, int]) -> float:
    tp = score_counts.get('TP', 0)
    fp = score_counts.get('FP', 0)
    tn = score_counts.get('TN', 0)
    fn = score_counts.get('FN', 0)
    prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    reca_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_pos = 2 * prec_pos * reca_pos / (prec_pos + reca_pos) if (prec_pos + reca_pos) > 0 else 0
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    reca_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_neg = 2 * prec_neg * reca_neg / (prec_neg + reca_neg) if (prec_neg + reca_neg) > 0 else 0
    macro_f1 = (f1_pos + f1_neg) / 2.0
    return macro_f1

# --- Data Loading ---
def load_dataset_csv_mapping(file_path: Union[str, Path], image_col: str, subset_col: str, label_col: Union[str, None] = None, real_label_value: Union[str, None] = None) -> Dict[str, str]:
    file_path = Path(file_path) # Ensure Path object
    if not file_path.exists(): # Use Path.exists()
        logger.warning(f"Data file not found: {file_path}")
        return {}
    try:
        data = pd.read_csv(file_path)
        subset_data = {}
        required_cols = [image_col, subset_col]
        if label_col: required_cols.append(label_col)

        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            logger.error(f"CSV {file_path} missing required columns: {missing}")
            return {}

        for _, row in data.iterrows():
            image_identifier = str(row[image_col]) # Ensure string
            raw_subset = 'unknown_subset'
            is_real = False
            if label_col and real_label_value and pd.notna(row[label_col]):
                if str(row[label_col]).lower() == real_label_value.lower():
                    is_real = True
            if is_real:
                raw_subset = real_label_value.lower() # Store lowercase 'real'
            elif pd.notna(row[subset_col]):
                raw_subset = str(row[subset_col]) # Ensure string
            else:
                logger.warning(f"Missing subset/generator value for image {image_identifier} in {file_path} and not identified as real.")
            subset_data[image_identifier] = raw_subset.lower()
        return subset_data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        return {}

def load_genimage_data_mapping(file_path: Union[str, Path]) -> Dict[str, str]:
    return load_dataset_csv_mapping(file_path, image_col='img_path', subset_col='dataset', label_col='dataset', real_label_value='real')

def load_df40_data_mapping(file_path: Union[str, Path]) -> Dict[str, str]:
    return load_dataset_csv_mapping(file_path, image_col='file_path', subset_col='dataset', label_col='label', real_label_value='real')

def load_d3_data_mapping(directory_path: Union[str, Path]) -> Dict[str, str]:
    dir_path = Path(directory_path) # Ensure Path object
    logger.info(f"Loading D3 image-to-subset mapping from directory: {dir_path}")
    if not dir_path.is_dir():
        logger.warning(f"D3 data directory not found: {dir_path}")
        return {}
    subset_data = {}
    processed_png_count = 0
    try:
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(".png"): # Case-insensitive
                processed_png_count += 1
                base_name = filename.rsplit('.', 1)[0] # More robust split for names with dots
                parts = base_name.split('_')
                raw_subset = 'unknown_d3_subset'
                if 'real' in parts[0].lower() and len(parts) >= 1: # Simpler real check
                    raw_subset = 'real'
                elif 'fake' in parts[0].lower() and len(parts) > 1:
                    potential_subset_parts = []
                    for part in parts[1:]:
                        if part.isdigit() or (part.startswith('gen') and part[3:].isdigit()):
                            break
                        potential_subset_parts.append(part)
                    if potential_subset_parts:
                        raw_subset = "_".join(potential_subset_parts)
                    elif len(parts) > 1: # Fallback: fake_subsetname
                        raw_subset = parts[1]
                else:
                    logger.warning(f"D3 file '{filename}' has an unrecognized naming pattern. Parts: {parts}. Assigning to '{raw_subset}'.")
                subset_data[filename] = str(raw_subset).lower()
    except Exception as e:
        logger.error(f"Error listing or processing D3 directory {dir_path}: {e}", exc_info=True)
        return {}
    logger.info(f"Processed {processed_png_count} PNG files for D3 mapping. Found {len(subset_data)} mappings.")
    return subset_data

def load_scores_from_jsonl_file(file_path: Union[str, Path]) -> Dict[str, int]:
    file_path = Path(file_path) # Ensure Path object
    scores = {}
    if not file_path.is_file():
        logger.error(f"Score file not found: {file_path}")
        return scores
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        all_records = []
        try:
            data = json.loads(content)
            if isinstance(data, list): all_records = data
            elif isinstance(data, dict) and 'rationales' in data and isinstance(data['rationales'], list):
                all_records = data['rationales'] # From scaling_consistency.py cache format
            elif isinstance(data, dict) and all(isinstance(item,dict) for item in data.get("rationales",[])): # from distinct_words.py format where item['rationales'] is text
                 all_records = data.get("rationales_data", data.get("rationales",[])) # Adjust based on actual distinct_words output structure
                 if not all_records and isinstance(data,list): # If data was a list of dicts from distinct_words
                     all_records = data

            else: # Try JSONL
                pass # Fall through
        except json.JSONDecodeError:
             logger.info(f"Could not parse {file_path} as single JSON. Assuming JSONL.", exc_info=True)
             # Fall through to JSONL parsing
        if not all_records:
            for line_num, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line: continue
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError as e_line:
                    logger.warning(f"Skipping line {line_num} in {file_path} (JSONL parse): {e_line}. Line: '{line[:100]}...'")
        if not all_records:
            logger.warning(f"No records found or parsed from {file_path}.")
            return scores
        for record_num, record in enumerate(all_records, 1):
            if not isinstance(record, dict):
                logger.warning(f"Record {record_num} in {file_path} is not a dict. Skipping.")
                continue
            image_path = record.get("image")
            cur_score_val = record.get("cur_score")
            if image_path is None or cur_score_val is None: continue
            if not isinstance(image_path, str): continue
            try:
                scores[image_path] = int(cur_score_val)
            except (ValueError, TypeError):
                logger.warning(f"Record {record_num} 'cur_score' ('{cur_score_val}') not int. Skipping.")
    except Exception as e:
        logger.error(f"Error reading or processing file {file_path}: {e}", exc_info=True)
    return scores


# --- Plotting & Color Utilities ---
# hex_to_rgb, rgb_to_hex, adjust_lightness, wordcloud_color_func_factory are mostly fine.
# Ensure logging for errors in these.
def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3: hex_color = ''.join([c*2 for c in hex_color])
        if len(hex_color) != 6: raise ValueError(f"Invalid hex color format: {hex_color}")
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError as e:
        logger.error(f"Hex to RGB conversion error: {e}", exc_info=True)
        return (0.0,0.0,0.0) # Return black on error


# --- Model Name Parsing ---
def parse_model_name(model_name_config: str) -> Tuple[str, str]:
    # This function seems mostly fine. Add logging for warnings.
    match = re.match(r'([a-zA-Z]+)(\d+(\.\d+)?)?-?(\d+[bBmB])', model_name_config)
    if match:
        base_name, version_with_dot, _, size_str = match.groups()
        version = version_with_dot if version_with_dot else ""
        family_name = base_name.lower()
        if family_name == 'qwen' and version == '25': family_name = 'qwen2.5'
        elif family_name == 'llama' and (version == '3' or version == '3.2' or not version):
             family_name = 'llama3.2' # Standardize to llama3.2
        elif version: family_name = f"{family_name}{version}"
        return family_name, size_str.lower()
    if model_name_config.upper() == "CODE": return "code", "6m" # Define standard size for CoDE
    logger.warning(f"Could not parse family/size reliably from LLM name: {model_name_config}. Returning as is.")
    return model_name_config.lower(), "unknown_size"

def load_rationales_from_file(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    file_path = Path(file_path)
    rationales_list = []
    if not file_path.is_file():
        logger.error(f"Rationale file not found: {file_path}")
        return rationales_list
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = json.load(f)
        if isinstance(content, list) and all(isinstance(item, dict) for item in content):
            rationales_list = content
        elif isinstance(content, dict) and 'rationales' in content and isinstance(content['rationales'], list):
            rationales_list = content['rationales']
        else:
            logger.error(f"Rationale file {file_path} is not a JSON list of dicts or expected dict structure. Type: {type(content)}")
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode JSON from rationale file {file_path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error reading rationale file {file_path}: {e}", exc_info=True)
    return rationales_list

def get_recall(true_positives: int, false_negatives: int) -> float:
    """Calculates recall: TP / (TP + FN)."""
    denominator = true_positives + false_negatives
    if denominator == 0:
        return 0.0
    return true_positives / denominator

# --- NEW HELPER FUNCTION for recall_subsets_table.py & combine_tables.py ---
def get_subset_recall_counts_from_rationales(
    rationale_file_path: Path,
    dataset_image_to_subset_map: Dict[str, str],
    dataset_name_for_logging: str = "unknown_dataset", # For better log messages
    image_key_in_json: str = 'image',
    score_key_in_json: str = 'cur_score',
    is_d3_dataset: bool = False # Special handling for D3 image keys
) -> Dict[str, Dict[str, int]]:
    """
    Processes a rationale file to count True Positives (TP) and False Negatives (FN)
    for each subset defined in the dataset_image_to_subset_map.

    Args:
        rationale_file_path: Path to the .jsonl file containing rationales.
        dataset_image_to_subset_map: Maps image identifiers (paths or filenames) to their true subset label.
        dataset_name_for_logging: Name of the dataset for logging purposes.
        image_key_in_json: Key in rationale records that holds the image identifier.
        score_key_in_json: Key in rationale records that holds the correctness score (1 for TP, 0 for FN,
                           assuming recall of the 'AI-generated' class or a similar binary task).
        is_d3_dataset: If True, the image identifier from the rationale file will be
                       processed with os.path.basename() to match keys in dataset_image_to_subset_map
                       (which are filenames for D3).

    Returns:
        A dictionary where keys are subset names and values are dicts {'TP': count, 'FN': count}.
    """
    subset_counts = defaultdict(lambda: {'TP': 0, 'FN': 0})
    
    # Load all rationale records from the file
    # This helper already handles JSON list vs JSONL
    rationale_records = load_rationales_from_file(rationale_file_path)
    if not rationale_records:
        logger.warning(f"No rationale records loaded from {rationale_file_path} for {dataset_name_for_logging}. Cannot calculate recall counts.")
        return dict(subset_counts) # Return empty if no records

    items_processed = 0
    items_skipped_no_mapping = 0
    items_skipped_bad_score = 0

    for record in rationale_records:
        if not isinstance(record, dict):
            logger.warning(f"Skipping non-dict record in {rationale_file_path}: {str(record)[:100]}")
            continue

        image_val = record.get(image_key_in_json)
        score_val = record.get(score_key_in_json)

        if image_val is None or score_val is None:
            logger.debug(f"Skipping record in {rationale_file_path} due to missing '{image_key_in_json}' or '{score_key_in_json}'. Record: {str(record)[:100]}")
            continue
        
        # Determine the lookup key for dataset_image_to_subset_map
        lookup_key = os.path.basename(str(image_val)) if is_d3_dataset else str(image_val)
        
        true_subset_lower = dataset_image_to_subset_map.get(lookup_key)

        if true_subset_lower is None:
            # This can be noisy if many images are not in the map, so consider logging level or sampling logs
            # logger.debug(f"Image key '{lookup_key}' (from value '{image_val}') not found in {dataset_name_for_logging} mapping. Skipping recall count for this item.")
            items_skipped_no_mapping += 1
            continue

        try:
            # Assuming score_val == 1 means the item was "correctly recalled" (e.g., AI image correctly identified as AI)
            # This corresponds to a True Positive for the class being recalled.
            # If score_val == 0, it was a "miss" for that class, hence a False Negative.
            if int(score_val) == 1:
                subset_counts[true_subset_lower]['TP'] += 1
            else:
                subset_counts[true_subset_lower]['FN'] += 1
            items_processed +=1
        except (ValueError, TypeError):
            logger.warning(f"Invalid score value '{score_val}' for key '{lookup_key}' in {rationale_file_path} for {dataset_name_for_logging}. Skipping recall count for this item.")
            items_skipped_bad_score +=1
            continue
            
    if items_skipped_no_mapping > 0:
        logger.info(f"For {dataset_name_for_logging} from {rationale_file_path}: {items_skipped_no_mapping} items skipped due to no mapping for image key.")
    if items_skipped_bad_score > 0:
        logger.warning(f"For {dataset_name_for_logging} from {rationale_file_path}: {items_skipped_bad_score} items skipped due to invalid score.")
    logger.info(f"Processed {items_processed} items for recall counts from {rationale_file_path} for {dataset_name_for_logging}.")
    
    return dict(subset_counts) # Convert back to regular dict for return


if __name__ == '__main__':
    # Example usage (can be expanded)
    logger.info("--- Helper Script Main Execution (for testing examples) ---")
    
    # Initialize environment (optional here, but good for testing seed-dependent things)
    # initialize_environment("0", 42)

    # Example for get_subset_recall_counts_from_rationales (requires dummy files and maps)
    # print("\n--- Testing get_subset_recall_counts_from_rationales ---")
    # dummy_rationale_file = Path("dummy_rationales.jsonl")
    # dummy_map = {"img1.png": "subset_a", "d3_images/img2.png": "subset_b", "img3.png": "subset_a"}
    # dummy_content = [
    #     {"image": "img1.png", "cur_score": 1},
    #     {"image": "d3_images/img2.png", "cur_score": 0}, # D3 path
    #     {"image": "img3.png", "cur_score": 1},
    #     {"image": "unknown.png", "cur_score": 1}, # Will be skipped
    #     {"image": "img1.png", "cur_score": "invalid_score"}, # Bad score
    # ]
    # try:
    #     with open(dummy_rationale_file, 'w') as f:
    #         for item in dummy_content:
    #             f.write(json.dumps(item) + "\n")
        
    #     print("Counts for non-D3:")
    #     counts_non_d3 = get_subset_recall_counts_from_rationales(dummy_rationale_file, dummy_map, "dummy_non_d3")
    #     print(json.dumps(counts_non_d3, indent=2))

    #     # Test D3 (keys in map should be basenames if is_d3_dataset=True)
    #     dummy_map_d3 = {"img2.png": "subset_b_d3"} # Map expects basename
    #     print("\nCounts for D3 (map expects basename):")
    #     counts_d3 = get_subset_recall_counts_from_rationales(dummy_rationale_file, dummy_map_d3, "dummy_d3", is_d3_dataset=True)
    #     print(json.dumps(counts_d3, indent=2))

    # finally:
    #     if dummy_rationale_file.exists():
    #         dummy_rationale_file.unlink()

    logger.info("--- Helper Script Main Execution Finished ---")