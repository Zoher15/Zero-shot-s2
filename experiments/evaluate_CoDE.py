"""
CoDE Model Evaluation Script

This script evaluates the CoDE (Contrastive Diffusion Estimation) model for AI-generated image detection.
CoDE is a supervised classification model that uses Vision Transformer features combined with a linear
classifier to distinguish between real and AI-generated images.

Unlike Vision-Language Models, CoDE:
- Does not use text prompts or instructions
- Directly classifies images without generating explanatory text
- Uses pre-trained ViT features with a linear classifier
- Returns binary predictions (real vs AI-generated)

The script loads images from datasets, processes them through the CoDE model pipeline,
and evaluates performance using accuracy and macro F1-score metrics.

Features:
- Batch processing for efficiency
- Robust error handling for image loading failures
- Progress tracking with tqdm
- Consistent output format compatible with other evaluation scripts
- Comprehensive logging and metrics collection

Usage:
    python experiments/evaluate_CoDE.py [options]
    
Examples:
    # Basic evaluation on DF40 2k dataset
    python experiments/evaluate_CoDE.py -d df402k -b 30
    
    # Evaluation on GenImage with larger batch size
    python experiments/evaluate_CoDE.py -d genimage2k -b 50 -c 0
    
Note:
    CoDE requires a CUDA-capable GPU for optimal performance. The model will
    automatically use CUDA if available, otherwise fall back to CPU.
"""

import sys
from pathlib import Path

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import config
from utils import helpers  # Main import for our helper functions
import argparse
import random  # Required for data shuffling

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="CoDE Model Evaluation Script for AI-Generated Image Detection")
parser.add_argument("-c", "--cuda", type=str, help="CUDA device IDs (e.g., '0' or '0,1')", default="0")
parser.add_argument("-d", "--dataset", type=str, help="Dataset to use (genimage2k, d32k, df402k)", default="df402k")
parser.add_argument("-b", "--batch_size", type=int, help="Batch size for model inference", default=30)

# Fixed attributes for CoDE model (not configurable via command line)
parser.llm = "CoDE"  # Model identifier for logging and output files
parser.mode = "direct_classification"  # Classification mode (no prompting)
parser.num = 1  # Number of sequences (always 1 for direct classification)
parser.prog = "evaluate_CoDE.py"  # Script identifier

args = parser.parse_args()

# --- Environment Initialization ---
helpers.initialize_environment(args.cuda)

# Import PyTorch and related libraries after environment setup
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import joblib  # For loading CoDE model's sklearn classifiers
import transformers
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import logging

# --- Logger Setup ---
helpers.setup_global_logger(config.EVAL_CODE_LOG_FILE)
# Get a logger instance for this specific module
logger = logging.getLogger(__name__)


# --- Data Loading Functions ---
def load_test_data_for_code(dataset_arg_val: str, question_str: str, current_config) -> list:
    """
    Load test data for CoDE evaluation from specified dataset.
    
    This function uses the same data loading utilities as VLM scripts but
    adapts them for CoDE's requirements. The question_str parameter is included
    for compatibility but is not used by CoDE for classification.
    
    Args:
        dataset_arg_val: Dataset identifier (e.g., 'genimage2k', 'd32k', 'df402k')
        question_str: Question phrase (included for compatibility, not used by CoDE)
        current_config: Configuration module containing dataset paths
        
    Returns:
        list: List of example dictionaries with 'image', 'answer', and 'question' fields
        
    Raises:
        SystemExit: If dataset is not recognized or loading fails
        
    Note:
        Data is shuffled with a fixed seed (0) to ensure reproducible results
        across multiple runs of the evaluation.
    """
    examples = []
    logger.info(f"Loading dataset for CoDE evaluation: {dataset_arg_val}")
    
    # Dispatch to appropriate dataset loader based on dataset name
    if 'genimage' in dataset_arg_val:
        file_to_load = current_config.GENIMAGE_2K_CSV_FILE if '2k' in dataset_arg_val else current_config.GENIMAGE_10K_CSV_FILE
        examples = helpers.load_genimage_data_examples(file_to_load, current_config.GENIMAGE_DIR, question_str)
    elif 'd3' in dataset_arg_val:
        examples = helpers.load_d3_data_examples(current_config.D3_DIR, question_str)
    elif 'df40' in dataset_arg_val:
        file_to_load = current_config.DF40_2K_CSV_FILE if '2k' in dataset_arg_val else current_config.DF40_10K_CSV_FILE
        examples = helpers.load_df40_data_examples(file_to_load, current_config.DF40_DIR, question_str)
    else:
        logger.error(f"Dataset '{dataset_arg_val}' not recognized for CoDE evaluation.")
        sys.exit(1)

    # Ensure reproducible data ordering across runs
    random.seed(0)
    random.shuffle(examples)
    logger.info(f"Loaded and shuffled {len(examples)} examples for dataset '{dataset_arg_val}'.")
    return examples


# --- Model Definition ---
class VITContrastiveHF(nn.Module):
    """
    CoDE Model Implementation using Hugging Face Vision Transformer.
    
    This model combines a pre-trained Vision Transformer with a linear classifier
    for AI-generated image detection. The architecture follows the CoDE paper
    implementation with feature extraction followed by sklearn classification.
    
    Attributes:
        model: Pre-trained Vision Transformer from Hugging Face
        processor: Image processor for preprocessing
        classifier: Pre-trained sklearn linear classifier
        
    Note:
        The model automatically downloads the pre-trained classifier weights
        from the Hugging Face model hub during initialization.
    """
    
    def __init__(self, repo_name, cache_dir=None):
        """
        Initialize the CoDE model with pre-trained components.
        
        Args:
            repo_name: Hugging Face repository name for the model
            cache_dir: Optional cache directory for model files
            
        Note:
            Downloads the sklearn classifier file 'linear_tot_classifier_epoch-32.sav'
            from the specified repository during initialization.
        """
        super(VITContrastiveHF, self).__init__()
        
        # Load pre-trained Vision Transformer
        self.model = transformers.AutoModel.from_pretrained(repo_name, cache_dir=cache_dir)
        self.model.pooler = nn.Identity()  # Remove pooling layer to access raw features
        
        # Load image processor (without resizing to preserve original resolution handling)
        self.processor = transformers.AutoProcessor.from_pretrained(repo_name, cache_dir=cache_dir)
        self.processor.do_resize = False
        
        # Download and load pre-trained sklearn classifier
        file_path = hf_hub_download(
            repo_id=repo_name, 
            filename='sklearn/linear_tot_classifier_epoch-32.sav', 
            cache_dir=cache_dir
        )
        self.classifier = joblib.load(file_path)

    def forward(self, x, return_feature=False):
        """
        Forward pass through the CoDE model.
        
        Args:
            x: Input tensor of preprocessed images
            return_feature: If True, return raw features; if False, return predictions
            
        Returns:
            torch.Tensor: Either raw features or classification predictions
            
        Note:
            Features are extracted from the [CLS] token (first token) of the
            transformer output, then passed through the sklearn classifier.
        """
        # Extract features using Vision Transformer
        features = self.model(x)
        
        if return_feature:
            return features
            
        # Extract [CLS] token features and move to CPU for sklearn
        features = features.last_hidden_state[:, 0, :].cpu().detach().numpy()
        
        # Get predictions from sklearn classifier
        predictions = self.classifier.predict(features)
        
        return torch.from_numpy(predictions)


# --- Model Initialization ---
logger.info("Loading CoDE model from Hugging Face...")
model = VITContrastiveHF(repo_name='aimagelab/CoDE').eval().to('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("CoDE model loaded successfully.")

# Define image preprocessing pipeline following CoDE requirements
transform = transforms.Compose([
    transforms.CenterCrop(224),  # Center crop to 224x224 (ViT standard)
    transforms.ToTensor(),       # Convert PIL to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])


# --- Inference Functions ---
def get_CoDE_response(example_batch, current_transform, current_model):
    """
    Generate CoDE model predictions for a batch of examples.
    
    Processes images through the preprocessing pipeline and CoDE model,
    handling any images that fail to load gracefully by skipping them.
    
    Args:
        example_batch: List of example dictionaries containing image paths
        current_transform: Image preprocessing transform pipeline
        current_model: Loaded CoDE model instance
        
    Returns:
        tuple: (pred_answers, processed_indices)
            - pred_answers: List of prediction strings ('real' or 'ai-generated')
            - processed_indices: List of indices of successfully processed images
            
    Note:
        CoDE model outputs:
        - 1: AI-generated image
        - 0: Real image  
        - -1: Abstained prediction (treated as real)
    """
    batch_of_images_tensors = []
    processed_indices = []  # Track which images were successfully processed
    
    # Process each image in the batch
    for idx, example in enumerate(example_batch):
        try:
            # Load and convert image to RGB
            img = Image.open(example['image']).convert('RGB')
            
            # Apply preprocessing transforms
            in_tens = current_transform(img)
            batch_of_images_tensors.append(in_tens)
            processed_indices.append(idx)
            
        except Exception as e:
            logger.warning(f"Error loading image {example.get('image', 'Unknown Image')}: {e}. Skipping.")
            # Continue processing other images in the batch
    
    # Return empty results if no images were successfully processed
    if not batch_of_images_tensors:
        return [], []

    # Stack images into batch tensor and move to model device
    input_batch_tensor = torch.stack(batch_of_images_tensors).to(current_model.model.device)
    
    # Generate predictions
    pred_answers = []
    with torch.no_grad():
        batch_predictions = current_model(input_batch_tensor).cpu().tolist()
    
    # Convert numerical predictions to string labels
    for pred in batch_predictions:
        if pred == 1:
            pred_answer = 'ai-generated'
        elif pred == 0 or pred == -1:  # 0 for real, -1 for abstained (treat as real)
            pred_answer = 'real'
        else:
            logger.warning(f"Unknown prediction value from CoDE: {pred}. Defaulting to 'unknown'.")
            pred_answer = 'unknown'  # Should not happen with proper sklearn classifier
        pred_answers.append(pred_answer)
        
    return pred_answers, processed_indices


# --- Main Evaluation Function ---
def eval_AI(current_model_str, test_data_list, current_batch_size, dataset_name_str):
    """
    Evaluate CoDE model on test dataset and compute performance metrics.
    
    Processes the test dataset in batches, collects predictions, computes
    accuracy and macro F1-score, and saves results in standard format.
    
    Args:
        current_model_str: Model identifier string (e.g., "CoDE")
        test_data_list: List of test examples with image paths and labels
        current_batch_size: Number of images to process per batch
        dataset_name_str: Dataset identifier for logging and output files
        
    Returns:
        float: Final macro F1-score (0-1 range)
        
    Note:
        Results are saved using the standard helpers.save_evaluation_outputs
        function to maintain consistency with VLM evaluation scripts.
        
    Confusion Matrix Convention:
        - TP: Correctly predicted real images
        - FN: Real images predicted as AI-generated  
        - TN: Correctly predicted AI-generated images
        - FP: AI-generated images predicted as real
    """
    logger.info(f"Starting CoDE evaluation: Model={current_model_str}, Dataset={dataset_name_str}, BatchSize={current_batch_size}")

    # Initialize evaluation metrics
    correct_count = 0
    confusion_matrix_counts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    rationales_data_output_list = []

    # Create batches for processing
    test_batches = [test_data_list[i:i + current_batch_size] for i in range(0, len(test_data_list), current_batch_size)]
    total_examples_processed_successfully = 0

    # Process batches with progress tracking
    with tqdm(total=len(test_data_list), dynamic_ncols=True) as pbar:
        for example_batch in test_batches:
            # Get model predictions for current batch
            pred_answers, processed_indices = get_CoDE_response(example_batch, transform, model)
            
            # Filter to successfully processed examples
            successfully_processed_examples_in_batch = [example_batch[i] for i in processed_indices]

            # Process each prediction
            for pred_idx, pred_answer in enumerate(pred_answers):
                example = successfully_processed_examples_in_batch[pred_idx]
                
                # Calculate correctness
                is_correct = 1 if pred_answer == example['answer'] else 0
                correct_count += is_correct
                total_examples_processed_successfully += 1

                # Update confusion matrix
                # Convention: 'real' is positive class, 'ai-generated' is negative class
                if example['answer'] == 'real':
                    if is_correct == 1:  # Predicted 'real', Ground truth 'real'
                        confusion_matrix_counts['TP'] += 1
                    else:  # Predicted 'ai-generated', Ground truth 'real'
                        confusion_matrix_counts['FN'] += 1
                elif example['answer'] == 'ai-generated':
                    if is_correct == 1:  # Predicted 'ai-generated', Ground truth 'ai-generated'
                        confusion_matrix_counts['TN'] += 1
                    else:  # Predicted 'real', Ground truth 'ai-generated'
                        confusion_matrix_counts['FP'] += 1
                
                # Calculate current macro F1-score
                current_macro_f1 = helpers.get_macro_f1_from_counts(confusion_matrix_counts)
                
                # Store detailed results for saving
                rationales_data_output_list.append({
                    "question": example.get('question', ""),  # Include question if present
                    "prompt": "",  # No prompt used for CoDE
                    "image": example['image'],
                    "rationales": [],  # No rationales generated by CoDE
                    'ground_answer': example['answer'],
                    'pred_answers': [pred_answer],  # List format for consistency
                    'pred_answer': pred_answer,     # Final prediction
                    'cur_score': is_correct
                })
                
                # Update progress bar
                if total_examples_processed_successfully > 0:
                    accuracy_val = correct_count / total_examples_processed_successfully
                else:
                    accuracy_val = 0
                
                pbar.set_description(
                    f"Macro-F1: {current_macro_f1*100:.2f}% || "
                    f"Accuracy: {accuracy_val*100:.2f}% "
                    f"({correct_count}/{total_examples_processed_successfully})"
                )
                pbar.update(1)  # Update progress by one processed item

    # Handle case where no examples were processed successfully
    if total_examples_processed_successfully == 0:
        logger.error("No examples were processed successfully. Cannot calculate final metrics.")
        return 0.0

    # Calculate final metrics
    final_macro_f1 = helpers.get_macro_f1_from_counts(confusion_matrix_counts)
    final_accuracy = correct_count / total_examples_processed_successfully
    
    logger.info(f"Evaluation completed: Accuracy={final_accuracy:.2%}, Macro F1={final_macro_f1:.3f}")
    
    # Save results using standard helper function
    helpers.save_evaluation_outputs(
        rationales_data=rationales_data_output_list,
        score_metrics=confusion_matrix_counts,
        macro_f1_score=final_macro_f1,  # Save as 0-1 range
        model_prefix="AI_CoDE",         # Prefix to distinguish from VLM results
        dataset_name=dataset_name_str,
        model_string=current_model_str,
        mode_type_str=args.mode,        # "direct_classification"
        num_sequences_val=args.num,     # 1 for single prediction
        config_module=config
    )
    
    return final_macro_f1


# --- Main Execution ---
if __name__ == "__main__":
    model_str_arg = "CoDE"  # Fixed model identifier for this script

    # Load test data using dataset-specific loaders
    logger.info(f"Loading dataset: {args.dataset}")
    question_phrase = config.EVAL_QUESTION_PHRASE  # For compatibility with data loaders
    images_test_data = load_test_data_for_code(args.dataset, question_phrase, config)

    # Validate that data was loaded successfully
    if not images_test_data:
        logger.error(f"Failed to load test data for dataset: {args.dataset}. Exiting.")
        sys.exit(1)
    
    logger.info(f"Successfully loaded {len(images_test_data)} examples for dataset '{args.dataset}'.")

    # Run evaluation
    final_f1_score = eval_AI(
        current_model_str=model_str_arg,
        test_data_list=images_test_data,
        current_batch_size=args.batch_size,
        dataset_name_str=args.dataset
    )

    # Log final results
    logger.info(f"CoDE evaluation completed for dataset: {args.dataset}")
    logger.info(f"Final Macro F1-Score: {final_f1_score*100:.2f}%")