import os
import sys
from pathlib import Path
import logging # Import the logging module

# --- Project Setup ---
# Ensures 'config' and 'utils' can be imported
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import config # For accessing configured paths like RESPONSES_DIR
from utils import helpers # For utility functions

# --- Logger Setup ---
helpers.setup_global_logger(config.RESULTS_FIND_IMAGES_LOG_FILE)
# Get a logger instance for this specific module.
logger = logging.getLogger(__name__)

# find_images_with_specific_scores function remains the same as its core logic is sound.
# We'll just ensure its print statements (if any were there) are also converted to logger calls.
def find_images_with_specific_scores(scores_m1, scores_m2, scores_m3):
    """
    Finds images where method1 (zeroshot) score is 0,
    method2 (zeroshot-cot) score is 0, and
    method3 (zeroshot-2-artifacts) score is 1.

    Args:
        scores_m1 (dict): Dictionary of image_path: score for method 1 (zeroshot).
        scores_m2 (dict): Dictionary of image_path: score for method 2 (zeroshot-cot).
        scores_m3 (dict): Dictionary of image_path: score for method 3 (zeroshot-2-artifacts).

    Returns:
        list: A list of image paths matching the criteria.
    """
    matching_images = []
    all_image_paths = set()

    # Consolidate image paths, ensuring keys are strings
    for scores_dict in [scores_m1, scores_m2, scores_m3]:
        if scores_dict: # Check if dictionary is not None or empty
            all_image_paths.update(k for k in scores_dict.keys() if isinstance(k, str))

    if not all_image_paths:
        logger.warning("No valid image paths found in any of the loaded scores to compare.")
        return matching_images

    processed_count = 0
    for image_path in all_image_paths:
        processed_count += 1
        score_m1 = scores_m1.get(image_path)
        score_m2 = scores_m2.get(image_path)
        score_m3 = scores_m3.get(image_path)

        # Check if all scores are available (not None) and match the criteria
        if score_m1 == 0 and score_m2 == 0 and score_m3 == 1:
            matching_images.append(image_path)

    logger.info(f"Processed {processed_count} unique image paths for score comparison.")
    return matching_images

def main():
    """
    Main function to execute the script.
    Uses helpers.load_scores_from_jsonl_file and logging.
    """
    logger.info("--- Starting Find Images Script ---")
    base_path = config.RESPONSES_DIR

    # --- Define file names for each method ---
    # These filenames should match exactly how they are saved by your evaluation scripts.
    # The original find_images.py used -wait0 in filenames.
    # The refactored eval scripts (via helpers.save_evaluation_outputs) likely save without -wait0 if wait is 0.
    # Adjust these filenames if needed.
    # Assuming n1 and wait0 are standard for these specific files:
    dataset_identifier = "df402k" # Example, make this configurable if needed
    model_identifier = "llama3-11b" # Example
    n_val = "n1"
    # wait_val_str = "wait0" # Only include if your filenames consistently have this

    # Construct filenames more robustly
    # If wait_val_str is part of your actual filenames:
    # file_name_method1 = f"AI_llama-{dataset_identifier}-{model_identifier}-zeroshot-{n_val}-{wait_val_str}-rationales.jsonl"
    # file_name_method2 = f"AI_llama-{dataset_identifier}-{model_identifier}-zeroshot-cot-{n_val}-{wait_val_str}-rationales.jsonl"
    # file_name_method3 = f"AI_llama-{dataset_identifier}-{model_identifier}-zeroshot-2-artifacts-{n_val}-{wait_val_str}-rationales.jsonl"

    # If -wait0 is omitted when wait is 0 (more likely with current eval script save logic):
    file_name_method1 = f"AI_llama-{dataset_identifier}-{model_identifier}-zeroshot-{n_val}-rationales.jsonl"
    file_name_method2 = f"AI_llama-{dataset_identifier}-{model_identifier}-zeroshot-cot-{n_val}-rationales.jsonl"
    file_name_method3 = f"AI_llama-{dataset_identifier}-{model_identifier}-zeroshot-2-artifacts-{n_val}-rationales.jsonl"


    file_path_method1 = base_path / file_name_method1
    file_path_method2 = base_path / file_name_method2
    file_path_method3 = base_path / file_name_method3
    # ---------------------------------------------

    logger.info(f"Attempting to load scores for Method 1 (zeroshot) from: {file_path_method1}")
    scores_method1 = helpers.load_scores_from_jsonl_file(file_path_method1)
    logger.info(f"Loaded {len(scores_method1)} scores for Method 1.")

    logger.info(f"Attempting to load scores for Method 2 (zeroshot-cot) from: {file_path_method2}")
    scores_method2 = helpers.load_scores_from_jsonl_file(file_path_method2)
    logger.info(f"Loaded {len(scores_method2)} scores for Method 2.")

    logger.info(f"Attempting to load scores for Method 3 (zeroshot-2-artifacts) from: {file_path_method3}")
    scores_method3 = helpers.load_scores_from_jsonl_file(file_path_method3)
    logger.info(f"Loaded {len(scores_method3)} scores for Method 3.")

    if not scores_method1 and not scores_method2 and not scores_method3:
        files_exist_check = [p.exists() for p in [file_path_method1, file_path_method2, file_path_method3]]
        if not any(files_exist_check):
            logger.critical("None of the specified data files were found. Please check paths and filenames.")
        else:
            logger.warning("No scores loaded from any files. Files might be empty or not in the expected format.")
        logger.info("Halting execution as no data is available to process.")
        return

    logger.info("Finding matching images based on score criteria...")
    result_images = find_images_with_specific_scores(scores_method1, scores_method2, scores_method3)

    if result_images:
        logger.info(f"Found {len(result_images)} images matching the criteria (zeroshot=0, zeroshot-cot=0, zeroshot-2-artifacts=1):")
        for image_path_result in result_images:
            # Using logger.info for each image path, could be print if it's direct user output.
            # For script output that might be parsed, logging is fine.
            logger.info(image_path_result)
    else:
        logger.info("No images found matching the specified criteria.")
    logger.info("--- Find Images Script Finished ---")

if __name__ == "__main__":
    main()