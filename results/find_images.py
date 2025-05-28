"""
Image Score Analysis Script

This script analyzes evaluation results to find images that demonstrate specific 
scoring patterns across different prompting methods. It identifies images where
standard zero-shot fails, chain-of-thought fails, but zero-shot-s² succeeds.

The script loads evaluation results from JSONL files and compares scores across
three prompting methods:
- Method 1: zeroshot (standard prompting)
- Method 2: zeroshot-cot (chain-of-thought prompting)  
- Method 3: zeroshot-2-artifacts (zero-shot-s² prompting)

Target Pattern:
- zeroshot score = 0 (incorrect prediction)
- zeroshot-cot score = 0 (incorrect prediction)
- zeroshot-2-artifacts score = 1 (correct prediction)

This pattern demonstrates cases where the zero-shot-s² method succeeds
when other methods fail, highlighting its effectiveness.

Usage:
    python results/find_images.py
    
Output:
    Logs matching image paths and summary statistics to console and log file.
    
Configuration:
    Modify dataset_identifier and model_identifier in main() to analyze
    different model/dataset combinations.
"""

import os
import sys
from pathlib import Path
import logging # Import the logging module

# --- Project Setup ---
# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import config # For accessing configured paths like RESPONSES_DIR
from utils import helpers # For utility functions

# --- Logger Setup ---
helpers.setup_global_logger(config.RESULTS_FIND_IMAGES_LOG_FILE)
# Get a logger instance for this specific module.
logger = logging.getLogger(__name__)

def find_images_with_specific_scores(scores_m1, scores_m2, scores_m3):
    """
    Find images matching specific score criteria across three prompting methods.
    
    Identifies images where Method 1 and Method 2 failed (score=0) but 
    Method 3 succeeded (score=1), demonstrating the effectiveness of the
    zero-shot-s² approach over standard and chain-of-thought prompting.
    
    Args:
        scores_m1 (dict): Dictionary mapping image_path -> score for Method 1 (zeroshot)
        scores_m2 (dict): Dictionary mapping image_path -> score for Method 2 (zeroshot-cot)
        scores_m3 (dict): Dictionary mapping image_path -> score for Method 3 (zeroshot-2-artifacts)
        
    Returns:
        list: List of image paths (strings) that match the criteria:
              - Method 1 score = 0 (incorrect)
              - Method 2 score = 0 (incorrect)  
              - Method 3 score = 1 (correct)
              
    Note:
        Only processes images that have scores available in all three method dictionaries.
        Images with missing scores in any method are excluded from the analysis.
    """
    matching_images = []
    all_image_paths = set()

    # Consolidate all image paths from the three score dictionaries
    for scores_dict in [scores_m1, scores_m2, scores_m3]:
        if scores_dict:  # Check if dictionary is not None or empty
            # Ensure keys are strings and add to consolidated set
            all_image_paths.update(k for k in scores_dict.keys() if isinstance(k, str))

    if not all_image_paths:
        logger.warning("No valid image paths found in any of the loaded score dictionaries.")
        return matching_images

    # Analyze each image path for the target score pattern
    processed_count = 0
    for image_path in all_image_paths:
        processed_count += 1
        
        # Get scores for this image from each method
        score_m1 = scores_m1.get(image_path)
        score_m2 = scores_m2.get(image_path)
        score_m3 = scores_m3.get(image_path)

        # Check if all scores are available and match the target criteria
        # Target: Method1=0, Method2=0, Method3=1 (both traditional methods fail, our method succeeds)
        if score_m1 == 0 and score_m2 == 0 and score_m3 == 1:
            matching_images.append(image_path)

    logger.info(f"Processed {processed_count} unique image paths for score comparison.")
    logger.info(f"Found {len(matching_images)} images matching the target pattern.")
    
    return matching_images

def main():
    """
    Main execution function for the image score analysis script.
    
    Loads evaluation results from JSONL files for three different prompting methods,
    analyzes score patterns, and reports images where zero-shot-s² succeeds
    when other methods fail.
    
    Configuration:
        Modify dataset_identifier and model_identifier variables to analyze
        different model/dataset combinations. The script constructs filenames
        based on the standard naming convention used by evaluation scripts.
        
    File Naming Convention:
        AI_{model_family}-{dataset}-{model}-{method}-{n_val}-rationales.jsonl
        
    Example Output Files Expected:
        - AI_llama-df402k-llama3-11b-zeroshot-n1-rationales.jsonl
        - AI_llama-df402k-llama3-11b-zeroshot-cot-n1-rationales.jsonl  
        - AI_llama-df402k-llama3-11b-zeroshot-2-artifacts-n1-rationales.jsonl
    """
    logger.info("=== Starting Image Score Analysis Script ===")
    
    # Base directory containing evaluation result files
    base_path = config.RESPONSES_DIR

    # --- Configuration: Modify these variables to analyze different combinations ---
    dataset_identifier = "df402k"      # Dataset to analyze (e.g., df402k, genimage2k, d32k)
    model_identifier = "llama3-11b"    # Model to analyze (e.g., llama3-11b, qwen25-7b)
    n_val = "n1"                       # Number of sequences (typically n1)
    
    logger.info(f"Analyzing results for dataset: {dataset_identifier}, model: {model_identifier}")

    # --- Construct filenames based on standard evaluation output naming ---
    # Standard format: AI_{model_family}-{dataset}-{model}-{method}-{n_val}-rationales.jsonl
    model_family = "llama" if "llama" in model_identifier.lower() else "qwen"
    
    file_name_method1 = f"AI_{model_family}-{dataset_identifier}-{model_identifier}-zeroshot-{n_val}-rationales.jsonl"
    file_name_method2 = f"AI_{model_family}-{dataset_identifier}-{model_identifier}-zeroshot-cot-{n_val}-rationales.jsonl"
    file_name_method3 = f"AI_{model_family}-{dataset_identifier}-{model_identifier}-zeroshot-2-artifacts-{n_val}-rationales.jsonl"

    # Construct full file paths
    file_path_method1 = base_path / file_name_method1
    file_path_method2 = base_path / file_name_method2
    file_path_method3 = base_path / file_name_method3

    # --- Load scores from each method's result file ---
    logger.info(f"Loading scores for Method 1 (zeroshot) from: {file_path_method1}")
    scores_method1 = helpers.load_scores_from_jsonl_file(file_path_method1)
    logger.info(f"Loaded {len(scores_method1)} scores for Method 1 (zeroshot).")

    logger.info(f"Loading scores for Method 2 (zeroshot-cot) from: {file_path_method2}")
    scores_method2 = helpers.load_scores_from_jsonl_file(file_path_method2)
    logger.info(f"Loaded {len(scores_method2)} scores for Method 2 (zeroshot-cot).")

    logger.info(f"Loading scores for Method 3 (zeroshot-2-artifacts) from: {file_path_method3}")
    scores_method3 = helpers.load_scores_from_jsonl_file(file_path_method3)
    logger.info(f"Loaded {len(scores_method3)} scores for Method 3 (zeroshot-2-artifacts).")

    # --- Validate that at least some data was loaded ---
    if not scores_method1 and not scores_method2 and not scores_method3:
        # Check if files exist to provide helpful error messages
        files_exist = [p.exists() for p in [file_path_method1, file_path_method2, file_path_method3]]
        existing_files = [str(p) for p, exists in zip([file_path_method1, file_path_method2, file_path_method3], files_exist) if exists]
        
        if not any(files_exist):
            logger.critical("None of the specified evaluation result files were found.")
            logger.critical("Expected files:")
            logger.critical(f"  - {file_path_method1}")
            logger.critical(f"  - {file_path_method2}")
            logger.critical(f"  - {file_path_method3}")
            logger.critical("Please run evaluations first or check file paths and naming convention.")
        else:
            logger.warning(f"Found {len(existing_files)} files but no scores were loaded.")
            logger.warning("Files might be empty or not in the expected JSONL format.")
            logger.warning(f"Existing files: {existing_files}")
            
        logger.info("Halting execution - no data available for analysis.")
        return

    # --- Analyze score patterns ---
    logger.info("Analyzing score patterns to find images where zero-shot-s² succeeds when others fail...")
    result_images = find_images_with_specific_scores(scores_method1, scores_method2, scores_method3)

    # --- Report results ---
    if result_images:
        logger.info(f"SUCCESS: Found {len(result_images)} images matching the criteria:")
        logger.info("  - zeroshot (Method 1) score = 0 (incorrect)")
        logger.info("  - zeroshot-cot (Method 2) score = 0 (incorrect)")  
        logger.info("  - zeroshot-2-artifacts (Method 3) score = 1 (correct)")
        logger.info("\nMatching image paths:")
        
        for i, image_path_result in enumerate(result_images, 1):
            logger.info(f"  {i:3d}. {image_path_result}")
            
        # Calculate percentage if we have total counts
        total_images = len(set().union(
            scores_method1.keys() if scores_method1 else [],
            scores_method2.keys() if scores_method2 else [],
            scores_method3.keys() if scores_method3 else []
        ))
        if total_images > 0:
            percentage = (len(result_images) / total_images) * 100
            logger.info(f"\nSummary: {len(result_images)}/{total_images} images ({percentage:.1f}%) show the target improvement pattern.")
    else:
        logger.info("No images found matching the specified criteria.")
        logger.info("This could indicate:")
        logger.info("  - All methods perform similarly on this dataset")
        logger.info("  - Different score criteria might be more appropriate")
        logger.info("  - Evaluation files might not contain expected score patterns")
    
    logger.info("=== Image Score Analysis Script Completed ===")

if __name__ == "__main__":
    main()