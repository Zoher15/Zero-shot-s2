"""
Utility Functions Module for Zero-shot-s² Repository

This module contains core utility functions used throughout the Zero-shot-s² project
for AI-generated image detection using Vision-Language Models. It provides essential
functionality for data loading, evaluation, logging, text processing, and results management.

The module is organized into the following functional areas:

## Logging and Environment Setup
- setup_global_logger(): Global logging configuration with file and console handlers
- initialize_environment(): CUDA environment setup and random seed configuration

## Evaluation and Validation
- validate_answers(): Extract and validate model predictions from responses
- update_progress(): Progress bar updates with accuracy and F1-score tracking
- save_evaluation_outputs(): Save evaluation results in standard formats (JSONL, JSON, CSV)
- get_macro_f1_from_counts(): Calculate macro F1-score from confusion matrix counts
- calculate_macro_f1_score_from_answers(): F1-score calculation from answer lists

## Data Loading Functions
- load_genimage_data_examples(): Load GenImage dataset samples
- load_d3_data_examples(): Load D3 dataset samples  
- load_df40_data_examples(): Load DF40 dataset samples
- load_dataset_csv_mapping(): Generic CSV-based dataset mapping loader
- load_rationales_from_file(): Load evaluation rationales from JSON/JSONL files

## Prompting and Model Configuration
- get_model_guiding_prefix_for_mode(): Generate model-specific prompt prefixes
- get_instructions_for_mode(): Get instruction text for different prompting modes
- parse_model_name(): Parse model names into family and size components

## Text Processing and NLP
- preprocess_text_to_token_set(): Text preprocessing with tokenization and lemmatization
- ensure_nltk_resources(): Ensure required NLTK resources are downloaded
- escape_latex(): Escape special LaTeX characters for table generation

## Results Processing and Analysis
- load_scores_from_jsonl_file(): Extract scores from evaluation result files
- load_scores_csv_to_dataframe(): Load CSV score files into DataFrames
- get_subset_recall_counts_from_rationales(): Calculate subset-specific recall metrics
- format_score_for_display(): Format numerical scores for display

## Utility Functions
- get_numeric_model_size(): Convert model size strings to numerical values
- get_recall(): Calculate recall from TP/FN counts
- hex_to_rgb(): Color utility for plotting

This module serves as the foundation for all evaluation and analysis scripts in the
Zero-shot-s² project, providing consistent interfaces for common operations and
ensuring reproducible results across different experimental configurations.

Usage:
    from utils import helpers
    
    # Set up logging
    helpers.setup_global_logger("experiment.log")
    
    # Load dataset
    examples = helpers.load_genimage_data_examples(csv_path, img_dir, question)
    
    # Process evaluation results
    f1_score = helpers.get_macro_f1_from_counts(confusion_matrix)
"""

# utils/helpers.py

import os
import sys
import json
import re
import string
from collections import Counter, defaultdict # defaultdict was missing from imports
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Union
import pandas as pd
from sklearn.metrics import f1_score
import nltk # Keep nltk import at top level
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging # Import logging
from tqdm import tqdm

# --- Standard Logger for this Module ---
logger = logging.getLogger(__name__)

def setup_global_logger(
    log_file_path_str: Union[str, Path],
    file_log_level: int = logging.INFO,
    console_log_level: int = logging.INFO,
    file_log_format: str = '%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
    console_log_format: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    file_log_mode: str = 'a',
    capture_stdout: bool = True
) -> None:
    """
    Enhanced global logger setup with stdout capture and distributed training support.
    
    Sets up a global logger that logs to a specified file and the console.
    Configures the root logger with these handlers, ensuring handlers are not duplicated.
    Enhanced version adopted from Zero-shot-mod for better logging capabilities.

    Args:
        log_file_path_str: Path to the log file.
        file_log_level: Logging level for the file handler.
        console_log_level: Logging level for the console handler.
        file_log_format: Log format for the file handler.
        console_log_format: Log format for the console handler.
        file_log_mode: File mode for the file handler (e.g., 'a' for append, 'w' for write).
        capture_stdout: Whether to capture stdout/stderr to the log file (enhanced feature).
    """
    log_file_path = Path(log_file_path_str)
    # Ensure the directory for the log file exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger() # Get the root logger

    # Set root logger level to the most verbose of the handlers if not already set,
    # or if current level is less verbose. This allows handlers to control their output.
    effective_root_level = min(file_log_level, console_log_level)
    if not root_logger.handlers or root_logger.level > effective_root_level:
        root_logger.setLevel(effective_root_level)

    # Configure File Handler for the root logger
    # Check if a FileHandler for this specific file already exists to prevent duplicates
    file_handler_exists = any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path)
        for h in root_logger.handlers
    )
    if not file_handler_exists:
        file_handler = logging.FileHandler(filename=str(log_file_path), mode=file_log_mode)
        file_handler.setLevel(file_log_level)
        file_formatter = logging.Formatter(file_log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Configure Console Handler (StreamHandler) for the root logger
    # Check if a StreamHandler to stdout already exists to prevent duplicates
    console_handler_exists = any(
        isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) == sys.stdout
        for h in root_logger.handlers
    )
    if not console_handler_exists:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_log_level)
        console_formatter = logging.Formatter(console_log_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

def initialize_environment(cuda_devices_str: str, seed_value: int = 0):
    """
    Initialize the experimental environment for reproducible ML experiments.
    
    Sets up CUDA device visibility, memory allocation configuration, and random seeds
    for PyTorch, NumPy, and Transformers to ensure reproducible results across runs.
    
    Args:
        cuda_devices_str: Comma-separated string of CUDA device IDs (e.g., "0", "0,1")
        seed_value: Random seed value for reproducibility (default: 0)
        
    Environment Variables Set:
        - CUDA_VISIBLE_DEVICES: Limits visible CUDA devices
        - PYTORCH_CUDA_ALLOC_CONF: Enables expandable memory segments
        
    Seeds Set:
        - Transformers library seed (for consistent model behavior)
        - PyTorch manual seed (CPU operations)
        - PyTorch CUDA seed (GPU operations, if available)
        - cuDNN deterministic mode (for consistent convolution algorithms)
        
    Note:
        This function should be called early in evaluation scripts before
        importing PyTorch or loading models to ensure proper initialization.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices_str
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Import transformers here to avoid module-level import before CUDA setup
    from transformers import set_seed
    set_seed(seed_value)
    
    import torch
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
    """
    Extract and validate model predictions from generated responses.
    
    Processes a list of model responses to extract predicted answers and rationales,
    using majority voting for the final prediction when multiple responses are provided.
    This function handles the standard answer format used in Zero-shot-s² evaluations.
    
    Args:
        example: Dictionary containing the ground truth example data (with 'answer' key)
        full_responses: List of raw text responses from the model
        labels: List of valid answer labels to search for (e.g., ['real', 'ai-generated'])
        answer_phrase: Delimiter phrase that separates rationale from final answer
                      (e.g., "Final Answer(real/ai-generated):")
    
    Returns:
        Tuple containing:
        - current_score (int): 1 if prediction matches ground truth, 0 otherwise
        - final_pred_answer (str): Final predicted answer after majority voting
        - pred_answers_extracted (List[str]): Individual predictions from each response
        - rationales_extracted (List[str]): Rationale text preceding each answer
        
    Processing Logic:
        1. For each response, split on answer_phrase to separate rationale and prediction
        2. Use regex matching to find valid labels in the prediction text
        3. Apply majority voting across all responses to determine final answer
        4. Compare final answer with ground truth (case-insensitive, parentheses removed)
        
    Example:
        response = "The image shows clear artifacts. Final Answer(real/ai-generated): ai-generated"
        # Extracts rationale: "The image shows clear artifacts."
        # Extracts prediction: "ai-generated"
    """
    ground_answer = example['answer'].lower().replace("(", "").replace(")", "")
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
    config_module: Any,  # Should be the imported config module
    **custom_params
):
    """
    Save evaluation results in standardized formats across the Zero-shot-s² project.
    
    Generates three output files for each evaluation run:
    1. Rationales file (JSONL): Detailed responses and predictions
    2. Scores file (JSON): Confusion matrix and detailed metrics  
    3. Scores file (CSV): Macro F1-score for easy aggregation
    
    Args:
        rationales_data: List of dictionaries containing detailed evaluation results.
                        Each dict should have keys: 'question', 'prompt', 'image', 
                        'rationales', 'ground_answer', 'pred_answers', 'pred_answer', 'cur_score'
        score_metrics: Dictionary with confusion matrix counts ('TP', 'FP', 'TN', 'FN')
        macro_f1_score: Final macro F1-score (0-1 range)
        model_prefix: Model family prefix (e.g., "AI_llama", "AI_qwen", "AI_CoDE")
        dataset_name: Dataset identifier (e.g., "genimage2k", "df402k", "d32k")
        model_string: Specific model name (e.g., "llama3-11b", "qwen25-7b")
        mode_type_str: Prompting mode (e.g., "zeroshot", "zeroshot-cot", "zeroshot-2-artifacts")
        num_sequences_val: Number of response sequences for self-consistency
        config_module: Imported config module providing RESPONSES_DIR and SCORES_DIR
        
    Output Files Generated:
        - {model_prefix}-{dataset_name}-{model_string}-{mode_type_str}-n{num_sequences_val}-rationales.jsonl
        - {model_prefix}-{dataset_name}-{model_string}-{mode_type_str}-n{num_sequences_val}-scores.json
        - {model_prefix}-{dataset_name}-{model_string}-{mode_type_str}-n{num_sequences_val}-scores.csv
        
    File Contents:
        - Rationales JSONL: Complete evaluation data for detailed analysis
        - Scores JSON: Confusion matrix and metrics for error analysis
        - Scores CSV: Single macro F1-score value for results aggregation
        
    Error Handling:
        Function handles directory creation failures and file I/O errors gracefully,
        logging errors while continuing to attempt saving other files.
    """
    # Ensure directories exist using enhanced directory management
    try:
        responses_dir = getattr(config_module, 'get_model_output_dir', None)
        if responses_dir and hasattr(config_module, 'get_filename'):
            # Use enhanced config functions
            responses_dir = config_module.get_model_output_dir(model_string, 'responses')
            scores_dir = config_module.get_model_output_dir(model_string, 'scores')
        else:
            # Fall back to original directories
            responses_dir = config_module.RESPONSES_DIR
            scores_dir = config_module.SCORES_DIR
            
        responses_dir.mkdir(parents=True, exist_ok=True)
        scores_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create output directories: {e}", exc_info=True)
        return # Cannot proceed if directories can't be made

    # Use enhanced filename generation if available
    if hasattr(config_module, 'get_filename'):
        rationales_filename = config_module.get_filename(
            "responses", dataset_name, model_string, mode_type_str, 
            num_sequences_val, **custom_params
        )
    else:
        # Fall back to original filename format
        base_filename_part = f"{model_prefix}-{dataset_name}-{model_string}-{mode_type_str}-n{num_sequences_val}"
        rationales_filename = f"{base_filename_part}-rationales.jsonl"
        
    rationales_file_path = responses_dir / rationales_filename
    try:
        with open(rationales_file_path, 'w', encoding='utf-8') as file: # Added encoding
            json.dump(rationales_data, file, indent=4)
        logger.info(f"Rationales data saved to {rationales_file_path}")
    except IOError as e:
        logger.error(f"Could not save rationales to {rationales_file_path}: {e}", exc_info=True)

    # Generate scores filename
    if hasattr(config_module, 'get_filename'):
        scores_filename_json = config_module.get_filename(
            "scores_json", dataset_name, model_string, mode_type_str, 
            num_sequences_val, **custom_params
        )
    else:
        base_filename_part = f"{model_prefix}-{dataset_name}-{model_string}-{mode_type_str}-n{num_sequences_val}"
        scores_filename_json = f"{base_filename_part}-scores.json"
        
    scores_file_path_json = scores_dir / scores_filename_json
    try:
        with open(scores_file_path_json, 'w', encoding='utf-8') as file: # Added encoding
            json.dump(score_metrics, file, indent=4)
        logger.info(f"Confusion matrix data (JSON) saved to {scores_file_path_json}")
    except IOError as e:
        logger.error(f"Could not save confusion matrix to {scores_file_path_json}: {e}", exc_info=True)

    csv_scores_dict = {f'{mode_type_str}-n{num_sequences_val}': macro_f1_score}
    scores_df = pd.DataFrame.from_dict(csv_scores_dict, orient='index', columns=['macro_f1'])

    # Generate CSV filename
    if hasattr(config_module, 'get_filename'):
        csv_scores_filename = config_module.get_filename(
            "scores_csv", dataset_name, model_string, mode_type_str, 
            num_sequences_val, **custom_params
        )
    else:
        base_filename_part = f"{model_prefix}-{dataset_name}-{model_string}-{mode_type_str}-n{num_sequences_val}"
        csv_scores_filename = f'{base_filename_part}-scores.csv'
        
    csv_file_path = scores_dir / csv_scores_filename
    try:
        scores_df.to_csv(csv_file_path, index=True, header=True)
        logger.info(f"Macro F1 score CSV saved to {csv_file_path}")
    except IOError as e:
        logger.error(f"Could not save Macro F1 CSV to {csv_file_path}: {e}", exc_info=True)

def get_model_guiding_prefix_for_mode(prompt_text: str, mode_type: str) -> str:
    prefixes = {
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
    prefix_to_add = ""
    for key_prefix, text_prefix in prefixes.items():
        if key_prefix in mode_type:
            prefix_to_add = text_prefix
            break
    if prefix_to_add: # Ensure space only if prefix is added
        return f"{prompt_text}{prefix_to_add}" # rstrip to avoid double spaces if prompt_text ends with one
    return prompt_text

def get_instructions_for_mode(mode_type: str) -> str:
    """
    Get system instructions based on the mode type.
    Returns empty string for most modes (no system instructions needed).
    """
    # Most modes don't require special system instructions
    # This can be extended if needed for specific modes
    instructions_map = {
        # Add any mode-specific system instructions here if needed
        # "special-mode": "You are an expert in...",
    }
    
    return instructions_map.get(mode_type, "")

# =============================================================================
# ENHANCED FILE LOADING FUNCTIONS
# =============================================================================
# Robust file loading with comprehensive error handling (adopted from Zero-shot-mod)

def load_file_safe(file_path: Union[str, Path], description: str = "file", is_jsonl: bool = False) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Safely load a JSON or JSONL file with proper error handling and validation.
    
    Enhanced version adopted from Zero-shot-mod for better error handling and validation.
    
    Args:
        file_path: Path to the file to load
        description: Human-readable description for error messages
        is_jsonl: Whether to treat the file as JSONL (line-delimited JSON)
        
    Returns:
        Loaded data (dict for JSON, list of dicts for JSONL)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file structure is invalid
        json.JSONDecodeError: If JSON is malformed
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path exists but is not a file: {file_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if is_jsonl:
                data = []
                for line_num, line in enumerate(f, 1):
                    if line.strip():  # Skip empty lines
                        try:
                            item = json.loads(line)
                            if not isinstance(item, dict):
                                logger.warning(f"Line {line_num} in {path} is not a JSON object, skipping")
                                continue
                            data.append(item)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num} in {path}: {e}, skipping")
                            continue
                
                if not data:
                    raise ValueError(f"No valid data found in {description}: {path}")
            else:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"{description} must contain a JSON object, got {type(data)}")
        
        logger.info(f"Loaded {description}: {path} ({len(data)} entries)")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {description} {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading {description} from {path}: {e}")
        raise

def load_genimage_data_examples(csv_file_path_str: Union[str, Path], image_base_dir_str: Union[str, Path], question_str: str) -> List[Dict[str, Any]]:
    csv_file_path = Path(csv_file_path_str)
    image_base_dir = Path(image_base_dir_str) # Use the provided image base directory
    data = pd.read_csv(csv_file_path)
    examples = []
    for _, row in data.iterrows():
        example_data = {}
        # Prepend the image_base_dir to the relative image path from the CSV
        example_data['image'] = str(image_base_dir / row['img_path'])
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

def load_df40_data_examples(csv_file_path_str: Union[str, Path], image_base_dir_str: Union[str, Path], question_str: str) -> List[Dict[str, Any]]:
    csv_file_path = Path(csv_file_path_str)
    image_base_dir = Path(image_base_dir_str) # Use the provided image base directory
    data = pd.read_csv(csv_file_path)
    examples = []
    for _, row in data.iterrows():
        example_data = {}
        example_data['image'] = str(image_base_dir / row['file_path'])
        example_data['question'] = question_str
        example_data['answer'] = 'real' if str(row['label']).lower() == 'real' else 'ai-generated'
        examples.append(example_data)
    return examples

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
        translator = str.maketrans('', '', string.punctuation + '’‘"”')
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

def get_numeric_model_size(size_str: str) -> float:
    """Converts a model size string (e.g., '7b', '6m') to a numeric value for sorting."""
    size_str = str(size_str).lower()
    if 'b' in size_str:
        return float(size_str.replace('b', '')) * 1e9
    if 'm' in size_str:
        return float(size_str.replace('m', '')) * 1e6
    try:
        return float(size_str) # For cases where size is just a number
    except ValueError:
        logger.warning(f"Could not convert size_str '{size_str}' to numeric. Returning inf.")
        return float('inf')

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

# Add to utils/helpers.py
def load_scores_csv_to_dataframe(score_file_path: Path) -> pd.DataFrame:
    if not score_file_path.is_file():
        logger.warning(f"Score CSV file not found: {score_file_path}")
        return pd.DataFrame() # Return empty DataFrame
    try:
        df = pd.read_csv(score_file_path, index_col=0) # Assumes index is method-n combo
        # The CSV typically has one column, e.g., 'macro_f1' or just values
        # The original script reads header=None, index_col=0, then df.loc[target_idx, 1]
        # This helper should probably just return the raw df.
        return df
    except Exception as e:
        logger.error(f"Error reading score CSV {score_file_path}: {e}", exc_info=True)
        return pd.DataFrame()

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

# =============================================================================
# ENHANCED UTILITY FUNCTIONS
# =============================================================================
# Additional utility functions adopted from Zero-shot-mod for better robustness

def validate_model_kwargs(model_kwargs: Dict[str, Any]) -> bool:
    """
    Validate model keyword arguments for generation.
    
    Enhanced validation function adopted from Zero-shot-mod to ensure
    generation parameters are within acceptable ranges and compatible.
    
    Args:
        model_kwargs: Dictionary of model generation parameters
        
    Returns:
        bool: True if all parameters are valid
        
    Raises:
        ValueError: If any parameters are invalid or incompatible
    """
    if not isinstance(model_kwargs, dict):
        raise ValueError(f"model_kwargs must be a dictionary, got {type(model_kwargs)}")
    
    # Validate numeric parameters
    numeric_params = {
        'max_new_tokens': (1, 8192),
        'temperature': (0.0, 2.0),
        'top_p': (0.0, 1.0),
        'top_k': (1, 1000),
        'repetition_penalty': (0.5, 2.0),
        'num_return_sequences': (1, 100)
    }
    
    for param, (min_val, max_val) in numeric_params.items():
        if param in model_kwargs:
            value = model_kwargs[param]
            if value is not None and (value < min_val or value > max_val):
                raise ValueError(f"{param} must be between {min_val} and {max_val}, got {value}")
    
    # Validate boolean parameters
    boolean_params = ['do_sample', 'return_dict_in_generate', 'output_scores']
    for param in boolean_params:
        if param in model_kwargs and not isinstance(model_kwargs[param], bool):
            raise ValueError(f"{param} must be a boolean, got {type(model_kwargs[param])}")
    
    # Validate dependencies
    if model_kwargs.get('do_sample') is True:
        if 'temperature' in model_kwargs and model_kwargs['temperature'] <= 0:
            raise ValueError("temperature must be > 0 when do_sample is True")
    
    logger.debug(f"Model kwargs validation passed: {model_kwargs}")
    return True

def clear_gpu_memory():
    """
    Clear GPU memory cache to prevent OOM errors.
    
    Enhanced memory management function adopted from Zero-shot-mod
    for better GPU memory handling during evaluation.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")
    except ImportError:
        logger.warning("PyTorch not available, cannot clear GPU memory")
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")

def get_default_generation_config() -> Dict[str, Any]:
    """
    Get default generation configuration with validation.
    
    Enhanced configuration function adopted from Zero-shot-mod patterns
    to provide consistent default generation parameters.
    
    Returns:
        Dict containing validated default generation parameters
    """
    config = {
        'max_new_tokens': 300,
        'do_sample': False,
        'temperature': 1.0,
        'top_p': None,
        'top_k': None,
        'repetition_penalty': 1.0,
        'num_return_sequences': 1
    }
    
    # Validate the default configuration
    validate_model_kwargs(config)
    
    return config

def ensure_directory_exists(directory_path: Union[str, Path], description: str = "directory") -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Enhanced directory management function adopted from Zero-shot-mod
    for robust directory handling with proper error reporting.
    
    Args:
        directory_path: Path to the directory
        description: Human-readable description for error messages
        
    Returns:
        Path: Absolute path to the directory
        
    Raises:
        ValueError: If path exists but is not a directory
        PermissionError: If directory cannot be created due to permissions
    """
    path = Path(directory_path).resolve()
    
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"{description} path exists but is not a directory: {path}")
    else:
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created {description}: {path}")
        except PermissionError as e:
            raise PermissionError(f"Cannot create {description} due to insufficient permissions: {path}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to create {description}: {path}") from e
    
    return path

def log_system_info():
    """
    Log system information for debugging and reproducibility.
    
    Enhanced system logging function adopted from Zero-shot-mod
    to provide better debugging and reproducibility information.
    """
    try:
        import platform
        import psutil
        
        logger.info("=== System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"CPU: {platform.processor()}")
        logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA: {torch.version.cuda}")
                logger.info(f"GPU: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            else:
                logger.info("CUDA: Not available")
        except ImportError:
            logger.info("PyTorch: Not available")
            
    except ImportError:
        logger.warning("System info packages not available")
    except Exception as e:
        logger.warning(f"Failed to log system info: {e}")
    finally:
        logger.info("=== End System Information ===")

def setup_evaluation_environment(cuda_devices: str, log_file: Union[str, Path], 
                                model_name: str = "unknown") -> None:
    """
    Comprehensive evaluation environment setup.
    
    Enhanced environment setup function that combines multiple initialization
    steps for consistent evaluation setup across scripts.
    
    Args:
        cuda_devices: CUDA device string (e.g., "0" or "0,1")
        log_file: Path to log file
        model_name: Name of model being evaluated
    """
    # Initialize environment and logging
    initialize_environment(cuda_devices)
    setup_global_logger(log_file, capture_stdout=True)
    
    # Log evaluation start
    logger.info(f"Starting evaluation with model: {model_name}")
    logger.info(f"CUDA devices: {cuda_devices}")
    
    # Log system information
    log_system_info()
    
    # Clear initial GPU memory
    clear_gpu_memory()

# =============================================================================
# VLM MODEL LOADING FUNCTIONS
# =============================================================================
# Vision-Language Model loading functions adopted from Zero-shot-mod patterns

def load_qwen_vlm_model(model_name: str):
    """
    Load Qwen Vision-Language Model with enhanced patterns.
    
    Args:
        model_name: Name of the Qwen VLM model
        
    Returns:
        Tuple of (model, processor)
        
    Raises:
        ValueError: If model name is not supported
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch
    
    # Get model configuration
    if model_name not in ['qwen25-3b', 'qwen25-7b', 'qwen25-32b', 'qwen25-72b']:
        available_models = ['qwen25-3b', 'qwen25-7b', 'qwen25-32b', 'qwen25-72b']
        raise ValueError(f"Unsupported Qwen model: {model_name}. Available: {available_models}")
    
    model_config = {
        "qwen25-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen25-7b": "Qwen/Qwen2.5-VL-7B-Instruct", 
        "qwen25-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
        "qwen25-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    }
    
    model_path = model_config[model_name]
    
    logger.info(f"Loading Qwen processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    
    logger.info(f"Loading Qwen model: {model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    
    logger.info("Compiling Qwen model...")
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        logger.info("Qwen model compilation complete.")
    except Exception as e:
        logger.warning(f"Qwen model compilation failed: {e}. Proceeding without compilation.")
    
    return model, processor

def load_llama_vlm_model(model_name: str):
    """
    Load Llama Vision-Language Model with enhanced patterns.
    
    Args:
        model_name: Name of the Llama VLM model
        
    Returns:
        Tuple of (model, processor)
        
    Raises:
        ValueError: If model name is not supported
    """
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    import torch
    
    llama_models = {
        "llama3-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "llama3-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",
    }
    
    if model_name not in llama_models:
        raise ValueError(f"Unsupported Llama model: {model_name}. Available: {list(llama_models.keys())}")
    
    model_path = llama_models[model_name]
    
    logger.info(f"Loading Llama processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    logger.info(f"Loading Llama model: {model_path}")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    
    logger.info("Compiling Llama model...")
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        logger.info("Llama model compilation complete.")
    except Exception as e:
        logger.warning(f"Llama model compilation failed: {e}. Proceeding without compilation.")
    
    return model, processor

def load_code_model():
    """
    Load CoDE computer vision model.
    
    Returns:
        Tuple of (model, processor) - both None for CoDE as it uses different approach
    """
    logger.info("Loading CoDE model...")
    # CoDE uses traditional computer vision approach, not a transformer model
    # This would initialize the CoDE model components
    return None, None

def load_vlm_model_unified(model_name: str):
    """
    Unified VLM model loader that handles all supported models.
    
    Args:
        model_name: Name of the VLM model to load
        
    Returns:
        Tuple of (model, processor, model_type) where model_type is 'qwen', 'llama', or 'code'
        
    Raises:
        ValueError: If model name is not supported
    """
    if model_name == "code":
        model, processor = load_code_model()
        return model, processor, 'code'
    elif model_name.startswith("qwen"):
        model, processor = load_qwen_vlm_model(model_name)
        return model, processor, 'qwen'
    elif model_name.startswith("llama"):
        model, processor = load_llama_vlm_model(model_name)
        return model, processor, 'llama'
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# =============================================================================
# VLM RESPONSE GENERATION FUNCTIONS
# =============================================================================
# Enhanced response generation functions for different VLM types

def generate_qwen_vlm_responses(model, processor, batch_examples: List[Dict], mode_type: str, num_sequences: int) -> List[str]:
    """
    Generate responses using Qwen VLM model with two-stage process.
    
    Args:
        model: Loaded Qwen model
        processor: Loaded Qwen processor
        batch_examples: List of examples with 'image' and 'question' keys
        mode_type: Prompting mode
        num_sequences: Number of sequences for self-consistency
        
    Returns:
        List of complete response strings
    """
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("qwen_vl_utils not available. Please install qwen-vl-utils for Qwen model support.")
    
    # Prepare prompts and messages
    instructions_text = get_instructions_for_mode(mode_type)
    prompt_texts = []
    messages_list = []
    
    for example in batch_examples:
        # Prepare prompt text
        prompt_text = example['question']
        if instructions_text:
            prompt_text = f"{instructions_text}\n\n{prompt_text}"
        
        # Add mode-specific prefix
        final_prompt_text = get_model_guiding_prefix_for_mode(prompt_text, mode_type)
        prompt_texts.append(final_prompt_text)
        
        # Store prompt in example for result saving
        example['prompt'] = final_prompt_text
        
        # Prepare messages
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": example['image']},
                {"type": "text", "text": example['question']}
            ]
        }]
        messages_list.append(messages)
    
    # Get generation parameters
    model_kwargs = get_generation_kwargs(num_sequences)
    
    # First generation
    image_inputs, video_inputs = process_vision_info(messages_list)
    image_inputs = [[image] for image in image_inputs]
    
    if len(prompt_texts) == 1:
        # Self-consistency mode
        k = model_kwargs.get('num_return_sequences', 1)
        prompts = prompt_texts * k
        final_image_inputs = image_inputs * k if image_inputs else None
    else:
        # Batch mode
        prompts = prompt_texts
        final_image_inputs = image_inputs
    
    # Process inputs for first generation
    processor_inputs = processor(
        text=prompts,
        images=final_image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    input_length = processor_inputs.input_ids.shape[1]
    
    # Generate first responses
    extra_args = {"return_dict_in_generate": True, "output_scores": True, "use_cache": True}
    model_kwargs_copy = model_kwargs.copy()
    model_kwargs_copy.pop('num_return_sequences', None)
    merged_args = {**processor_inputs, **model_kwargs_copy, **extra_args}
    
    with torch.no_grad():
        outputs = model.generate(**merged_args)
    
    first_responses = processor.batch_decode(
        outputs.sequences[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    # Memory cleanup
    del processor_inputs, outputs
    clear_gpu_memory()
    
    # Second generation for answer formatting
    answer_phrase = "Final Answer(real/ai-generated):"
    first_cut_responses = [resp.split(answer_phrase)[0] for resp in first_responses]
    
    if len(prompt_texts) == 1:
        # Self-consistency mode
        second_prompts = [f"{prompt_texts[0]}{cut_resp} {answer_phrase}"
                         for cut_resp in first_cut_responses]
        final_image_inputs = image_inputs * len(second_prompts) if image_inputs else None
    else:
        # Batch mode
        second_prompts = [f"{prompt_texts[i]}{first_cut_responses[i]} {answer_phrase}"
                         for i in range(len(first_cut_responses))]
        final_image_inputs = image_inputs
    
    # Process inputs for second generation
    processor_inputs = processor(
        text=second_prompts,
        images=final_image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    input_length = processor_inputs.input_ids.shape[1]
    
    # Generate second responses
    model_kwargs_copy['do_sample'] = False
    merged_args = {**processor_inputs, **model_kwargs_copy, **extra_args}
    
    with torch.no_grad():
        outputs = model.generate(**merged_args)
    
    second_responses = processor.batch_decode(
        outputs.sequences[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    # Combine first and second responses
    full_responses = [f"{first_cut_responses[i]} {answer_phrase}{second_responses[i]}"
                     for i in range(len(second_responses))]
    
    # Memory cleanup
    del processor_inputs, outputs
    clear_gpu_memory()
    
    return full_responses

def generate_llama_vlm_responses(model, processor, batch_examples: List[Dict], mode_type: str, num_sequences: int) -> List[str]:
    """
    Generate responses using Llama VLM model with two-stage process.
    
    Args:
        model: Loaded Llama model
        processor: Loaded Llama processor
        batch_examples: List of examples with 'image' and 'question' keys
        mode_type: Prompting mode
        num_sequences: Number of sequences for self-consistency
        
    Returns:
        List of complete response strings
    """
    from PIL import Image
    
    # Prepare prompts
    instructions_text = get_instructions_for_mode(mode_type)
    prompt_texts = []
    image_paths = []
    
    for example in batch_examples:
        messages = []
        if instructions_text:
            messages.append({
                "role": "system", 
                "content": [{"type": "text", "text": instructions_text}]
            })
        messages.append({
            "role": "user", 
            "content": [{"type": "image"}, {"type": "text", "text": example['question']}]
        })
        
        prompt_text = processor.apply_chat_template(
            messages, 
            padding=False, 
            tokenize=False, 
            truncation=True, 
            add_generation_prompt=True
        )
        
        # Add mode-specific prefix
        final_prompt_text = get_model_guiding_prefix_for_mode(prompt_text, mode_type)
        prompt_texts.append(final_prompt_text)
        image_paths.append(example['image'])
        
        # Store prompt in example for result saving
        example['prompt'] = final_prompt_text
    
    # Get generation parameters
    model_kwargs = get_generation_kwargs(num_sequences)
    
    # First generation
    image_inputs = [[Image.open(path).convert("RGB")] for path in image_paths]
    
    if len(prompt_texts) == 1:
        # Self-consistency mode
        k = model_kwargs.get('num_return_sequences', 1)
        prompts = prompt_texts * k
        final_image_inputs = image_inputs * k
    else:
        # Batch mode
        prompts = prompt_texts
        final_image_inputs = image_inputs
    
    # Process inputs for first generation
    processor_inputs = processor(
        text=prompts,
        images=final_image_inputs,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)
    
    input_length = processor_inputs.input_ids.shape[1]
    
    # Generate first responses
    extra_args = {"return_dict_in_generate": True, "output_scores": True, "use_cache": True}
    model_kwargs_copy = model_kwargs.copy()
    model_kwargs_copy.pop('num_return_sequences', None)
    merged_args = {**processor_inputs, **model_kwargs_copy, **extra_args}
    
    with torch.no_grad():
        outputs = model.generate(**merged_args)
    
    first_responses = processor.batch_decode(
        outputs.sequences[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    # Memory cleanup
    del processor_inputs, outputs
    clear_gpu_memory()
    
    # Second generation for answer formatting
    answer_phrase = "Final Answer(real/ai-generated):"
    first_cut_responses = [resp.split(answer_phrase)[0] for resp in first_responses]
    
    if len(prompt_texts) == 1:
        # Self-consistency mode
        second_prompts = [f"{prompt_texts[0]}{cut_resp} {answer_phrase}"
                         for cut_resp in first_cut_responses]
        final_image_inputs = image_inputs * len(second_prompts)
    else:
        # Batch mode
        second_prompts = [f"{prompt_texts[i]}{first_cut_responses[i]} {answer_phrase}"
                         for i in range(len(first_cut_responses))]
        final_image_inputs = image_inputs
    
    # Process inputs for second generation
    processor_inputs = processor(
        text=second_prompts,
        images=final_image_inputs,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)
    
    input_length = processor_inputs.input_ids.shape[1]
    
    # Generate second responses
    model_kwargs_copy['do_sample'] = False
    merged_args = {**processor_inputs, **model_kwargs_copy, **extra_args}
    
    with torch.no_grad():
        outputs = model.generate(**merged_args)
    
    second_responses = processor.batch_decode(
        outputs.sequences[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    # Combine first and second responses
    full_responses = [f"{first_cut_responses[i]} {answer_phrase}{second_responses[i]}"
                     for i in range(len(second_responses))]
    
    # Memory cleanup
    del processor_inputs, outputs
    clear_gpu_memory()
    
    return full_responses

def generate_code_responses(batch_examples: List[Dict], num_sequences: int) -> List[str]:
    """
    Generate responses using CoDE computer vision model.
    
    Args:
        batch_examples: List of examples with 'image' and 'question' keys
        num_sequences: Number of sequences for self-consistency
        
    Returns:
        List of response strings
    """
    # TODO: Implement CoDE response generation
    # This would use traditional computer vision techniques to analyze images
    responses = []
    answer_phrase = "Final Answer(real/ai-generated):"
    
    for example in batch_examples:
        # Placeholder - would analyze image using computer vision techniques
        # For now, return a simple response
        response = f"{answer_phrase} ai-generated"
        responses.extend([response] * num_sequences)
    
    return responses

def create_vlm_response_generator_unified(model, processor, model_type: str, mode_type: str):
    """
    Create a unified response generator function for VLM models.
    
    This function creates a closure that captures the model, processor, and mode
    and returns a function compatible with the shared evaluation logic.
    
    Args:
        model: Loaded VLM model
        processor: Loaded VLM processor  
        model_type: Type of model ('qwen', 'llama', 'code')
        mode_type: Prompting mode
        
    Returns:
        Function that generates responses for batches
    """
    def generate_responses(batch_examples: List[Dict], batch_size: int, num_sequences: int) -> List[str]:
        """Generate responses for a batch of examples."""
        clear_gpu_memory()
        
        if model_type == 'code':
            return generate_code_responses(batch_examples, num_sequences)
        elif model_type == 'qwen':
            return generate_qwen_vlm_responses(model, processor, batch_examples, mode_type, num_sequences)
        elif model_type == 'llama':
            return generate_llama_vlm_responses(model, processor, batch_examples, mode_type, num_sequences)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    return generate_responses

# =============================================================================
# SHARED MODEL EVALUATION FUNCTIONS
# =============================================================================

def get_generation_kwargs(num_sequences: int) -> Dict[str, Any]:
    """
    Get standard generation parameters for VLM evaluation.
    
    Args:
        num_sequences: Number of sequences for self-consistency
        
    Returns:
        Dictionary of generation parameters
    """
    if num_sequences == 1:
        return {
            "max_new_tokens": 300,
            "do_sample": False,
            "repetition_penalty": 1,
            "top_k": None,
            "top_p": None,
            "temperature": 1
        }
    else:
        return {
            "max_new_tokens": 300,
            "do_sample": True,
            "repetition_penalty": 1,
            "top_k": None,
            "top_p": None,
            "temperature": 1,
            "num_return_sequences": num_sequences
        }

def run_model_evaluation(
    test_data: List[Dict[str, Any]],
    response_generator_fn,
    model_name: str,
    mode_type: str,
    num_sequences: int,
    batch_size: int,
    dataset_name: str,
    model_prefix: str,
    config_module: Any
) -> float:
    """
    Unified evaluation loop for VLM models.
    
    This function handles the core evaluation logic that's common to all VLM models,
    including metrics tracking, progress updates, and result saving.
    
    Args:
        test_data: List of test examples
        response_generator_fn: Function that generates responses for a batch
                              Signature: fn(batch_examples, batch_size, num_sequences) -> List[str]
        model_name: Name of the model being evaluated
        mode_type: Prompting mode
        num_sequences: Number of sequences for self-consistency
        batch_size: Batch size for inference
        dataset_name: Dataset identifier
        model_prefix: Model prefix for output files (e.g., "AI_llama")
        config_module: Configuration module
        
    Returns:
        Final macro F1-score
    """
    logger.info(f"Starting evaluation: Model={model_name}, Dataset={dataset_name}, "
               f"Mode={mode_type}, NumSequences={num_sequences}, BatchSize={batch_size}")
    
    # Initialize evaluation metrics
    correct_count = 0
    confusion_matrix_counts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    rationales_data_output_list = []
    labels_for_validation = ['ai-generated', 'real']
    answer_phrase = config_module.EVAL_ANSWER_PHRASE
    
    # Determine actual batch size (self-consistency uses batch_size=1)
    actual_batch_size = 1 if num_sequences > 1 else batch_size
    
    # Process in batches
    with tqdm(total=len(test_data), dynamic_ncols=True) as pbar:
        for i in range(0, len(test_data), actual_batch_size):
            # Get current batch
            batch_examples = test_data[i:i + actual_batch_size]
            
            # Generate responses using model-specific function
            full_responses = response_generator_fn(batch_examples, actual_batch_size, num_sequences)
            
            # Process responses and update metrics
            if actual_batch_size == 1 and num_sequences > 1:
                # Self-consistency mode: single example with multiple responses
                example = batch_examples[0]
                
                cur_score, pred_answer, pred_answers_list, rationales_list = validate_answers(
                    example, full_responses, labels_for_validation, answer_phrase
                )
                
                # Update metrics
                correct_count += cur_score
                if example['answer'] == 'real':
                    if cur_score == 1:
                        confusion_matrix_counts['TP'] += 1
                    else:
                        confusion_matrix_counts['FN'] += 1
                elif example['answer'] == 'ai-generated':
                    if cur_score == 1:
                        confusion_matrix_counts['TN'] += 1
                    else:
                        confusion_matrix_counts['FP'] += 1
                
                # Store detailed results
                current_macro_f1 = get_macro_f1_from_counts(confusion_matrix_counts)
                rationales_data_output_list.append({
                    "question": example['question'],
                    "prompt": example.get('prompt', ''),  # May be set by response generator
                    "image": example['image'],
                    "rationales": rationales_list,
                    'ground_answer': example['answer'],
                    'pred_answers': pred_answers_list,
                    'pred_answer': pred_answer,
                    'cur_score': cur_score
                })
                
                update_progress(pbar, correct_count, current_macro_f1 * 100)
                
            else:
                # Batch mode: multiple examples with single response each
                for idx, single_response in enumerate(full_responses):
                    example = batch_examples[idx]
                    
                    cur_score, pred_answer, pred_answers_list, rationales_list = validate_answers(
                        example, [single_response], labels_for_validation, answer_phrase
                    )
                    
                    # Update metrics
                    correct_count += cur_score
                    if example['answer'] == 'real':
                        if cur_score == 1:
                            confusion_matrix_counts['TP'] += 1
                        else:
                            confusion_matrix_counts['FN'] += 1
                    elif example['answer'] == 'ai-generated':
                        if cur_score == 1:
                            confusion_matrix_counts['TN'] += 1
                        else:
                            confusion_matrix_counts['FP'] += 1
                    
                    # Store detailed results
                    current_macro_f1 = get_macro_f1_from_counts(confusion_matrix_counts)
                    rationales_data_output_list.append({
                        "question": example['question'],
                        "prompt": example.get('prompt', ''),  # May be set by response generator
                        "image": example['image'],
                        "rationales": rationales_list,
                        'ground_answer': example['answer'],
                        'pred_answers': pred_answers_list,
                        'pred_answer': pred_answer,
                        'cur_score': cur_score
                    })
                    
                    update_progress(pbar, correct_count, current_macro_f1 * 100)
    
    # Calculate final metrics and save results
    final_macro_f1 = get_macro_f1_from_counts(confusion_matrix_counts)
    
    save_evaluation_outputs(
        rationales_data_output_list, 
        confusion_matrix_counts, 
        final_macro_f1,
        model_prefix, 
        dataset_name, 
        model_name, 
        mode_type, 
        num_sequences, 
        config_module
    )
    
    return final_macro_f1

def load_test_data_unified(dataset_arg_val: str, question_str: str, config_module: Any) -> List[Dict[str, Any]]:
    """
    Load test data for evaluation using dataset-specific helpers.
    
    Args:
        dataset_arg_val: Dataset identifier (e.g., 'genimage2k', 'd32k', 'df402k')
        question_str: Question text to include in examples
        config_module: Configuration module containing dataset paths
        
    Returns:
        List of example dictionaries with 'image', 'question', 'answer' keys
    """
    examples = []
    logger.info(f"Loading dataset: {dataset_arg_val}")
    
    if 'genimage' in dataset_arg_val:
        file_to_load = config_module.GENIMAGE_2K_CSV_FILE if '2k' in dataset_arg_val else config_module.GENIMAGE_10K_CSV_FILE
        examples = load_genimage_data_examples(file_to_load, config_module.GENIMAGE_DIR, question_str)
    elif 'd3' in dataset_arg_val:
        examples = load_d3_data_examples(config_module.D3_DIR, question_str)
    elif 'df40' in dataset_arg_val:
        file_to_load = config_module.DF40_2K_CSV_FILE if '2k' in dataset_arg_val else config_module.DF40_10K_CSV_FILE
        examples = load_df40_data_examples(file_to_load, config_module.DF40_DIR, question_str)
    else:
        logger.error(f"Dataset '{dataset_arg_val}' not recognized.")
        return []
    
    # Shuffle for consistent evaluation
    import random
    random.seed(0)
    random.shuffle(examples)
    logger.info(f"Loaded and shuffled {len(examples)} examples for dataset '{dataset_arg_val}'.")
    
    return examples

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