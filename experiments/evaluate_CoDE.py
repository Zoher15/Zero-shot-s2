import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import argparse # Keep argparse for specific CoDE arguments if not in helpers
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import joblib # For loading CoDE model classifiers
import transformers
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import config
from utils import helpers
import logging

# --- Logger Setup ---
helpers.setup_global_logger(config.EVAL_CODE_LOG_FILE)
# Get a logger instance for this specific module.
logger = logging.getLogger(__name__)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Vision-Language Model Evaluation Script")
parser.add_argument("-c", "--cuda", type=str, help="CUDA device IDs (e.g., '0' or '0,1')", default="0")
parser.add_argument("-d", "--dataset", type=str, help="Dataset to use (e.g., 'genimage2k')", default="df402k")
parser.add_argument("-b", "--batch_size", type=int, help="Batch size for model inference", default=30)
parser.llm = "CoDE" # Default for CoDE, not used in the model
parser.mode = "direct_classification" # Default for CoDE, not used in the model
parser.num = 1 # Default for CoDE, not used in the model
parser.prog = "evaluate_CoDE.py"
args = parser.parse_args()

# --- Environment Initialization ---
helpers.initialize_environment(args.cuda) # Default seed 0 is handled by initialize_environment

# --- Data Loading Dispatcher ---
def load_test_data_for_code(dataset_arg_val: str, question_str: str, current_config) -> list:
    examples = []
    logger.info(f"Attempting to load dataset for CoDE: {dataset_arg_val}") # Assuming logger is set up
    if 'genimage' in dataset_arg_val:
        file_to_load = current_config.GENIMAGE_2K_CSV_FILE if '2k' in dataset_arg_val else current_config.GENIMAGE_10K_CSV_FILE
        examples = helpers.load_genimage_data_examples(file_to_load, current_config.GENIMAGE_DIR, question_str)
    elif 'd3' in dataset_arg_val:
        examples = helpers.load_d3_data_examples(current_config.D3_DIR, question_str)
    elif 'df40' in dataset_arg_val:
        file_to_load = current_config.DF40_2K_CSV_FILE if '2k' in dataset_arg_val else current_config.DF40_10K_CSV_FILE
        examples = helpers.load_df40_data_examples(file_to_load, current_config.DF40_DIR, question_str)
    # Add other dataset conditions if CoDE supports them, similar to helpers.load_test_data
    else:
        logger.error(f"Dataset '{dataset_arg_val}' not recognized for path configuration in CoDE script.")
        sys.exit(1) # Or raise an error

    random.seed(0) # Ensure consistent shuffle
    random.shuffle(examples)
    logger.info(f"Loaded and shuffled {len(examples)} examples for dataset '{dataset_arg_val}'.")
    return examples

# --- Model Definition and Initialization ---
class VITContrastiveHF(nn.Module):
    def __init__(self, repo_name, cache_dir=None):
        super(VITContrastiveHF, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(repo_name, cache_dir=cache_dir)
        self.model.pooler = nn.Identity()
        self.processor = transformers.AutoProcessor.from_pretrained(repo_name, cache_dir=cache_dir)
        self.processor.do_resize = False
        file_path = hf_hub_download(repo_id=repo_name, filename='sklearn/linear_tot_classifier_epoch-32.sav', cache_dir=cache_dir)
        self.classifier = joblib.load(file_path)

    def forward(self, x, return_feature=False):
        features = self.model(x)
        if return_feature:
            return features
        features = features.last_hidden_state[:, 0, :].cpu().detach().numpy()
        predictions = self.classifier.predict(features)
        return torch.from_numpy(predictions)

logger.info("Loading CoDE model...")
model = VITContrastiveHF(repo_name='aimagelab/CoDE').eval().to('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("CoDE model loaded.")

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Model Response Generation ---
def get_CoDE_response(example_batch, current_transform, current_model):
    batch_of_images_tensors = []
    processed_indices = [] # Keep track of which images were successfully processed
    for idx, example in enumerate(example_batch):
        try:
            img = Image.open(example['image']).convert('RGB')
            in_tens = current_transform(img)
            batch_of_images_tensors.append(in_tens)
            processed_indices.append(idx)
        except Exception as e:
            logger.warning(f"Error loading image {example.get('image', 'Unknown Image')}: {e}. Skipping this image.")
            # We'll return fewer predictions than the input batch size if an image fails
    
    if not batch_of_images_tensors:
        return [], [] # Return empty if no images were processed

    input_batch_tensor = torch.stack(batch_of_images_tensors).to(current_model.model.device) # Use model's device
    
    pred_answers = []
    with torch.no_grad():
        batch_predictions = current_model(input_batch_tensor).cpu().tolist()
    
    for pred in batch_predictions:
        if pred == 1:
            pred_answer = 'ai-generated'
        elif pred == 0 or pred == -1: # CoDE paper mentions 0 for real, -1 for abstained (treat as real)
            pred_answer = 'real'
        else:
            logger.warning(f"Unknown prediction value from CoDE: {pred}. Defaulting to 'unknown'.")
            pred_answer = 'unknown' # Should ideally not happen with the sklearn classifier
        pred_answers.append(pred_answer)
        
    return pred_answers, processed_indices


# --- Main Evaluation Logic ---
def eval_AI(current_model_str, test_data_list, current_batch_size, dataset_name_str):
    logger.info(f"Starting CoDE evaluation: Model={current_model_str}, Dataset={dataset_name_str}, BatchSize={current_batch_size}")

    correct_count = 0
    # Using the confusion matrix structure expected by helpers.get_macro_f1_from_counts
    confusion_matrix_counts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    rationales_data_output_list = []

    # Create batches
    test_batches = [test_data_list[i:i + current_batch_size] for i in range(0, len(test_data_list), current_batch_size)]
    
    total_examples_processed_successfully = 0

    with tqdm(total=len(test_data_list), dynamic_ncols=True) as pbar:
        for example_batch in test_batches:
            pred_answers, processed_indices = get_CoDE_response(example_batch, transform, model)
            
            successfully_processed_examples_in_batch = [example_batch[i] for i in processed_indices]

            for pred_idx, pred_answer in enumerate(pred_answers):
                example = successfully_processed_examples_in_batch[pred_idx]
                
                # CoDE does not use a "question", so we use a placeholder or omit.
                # Rationales are also not generated by CoDE.
                # The 'prompt' field is also not applicable.
                
                is_correct = 1 if pred_answer == example['answer'] else 0
                correct_count += is_correct
                total_examples_processed_successfully +=1

                # Update confusion matrix based on ground truth and prediction
                # Assuming 'real' is the positive class (TP, FN) and 'ai-generated' is the negative class (TN, FP)
                if example['answer'] == 'real':
                    if is_correct == 1: # Predicted 'real', Ground 'real'
                        confusion_matrix_counts['TP'] += 1
                    else: # Predicted 'ai-generated', Ground 'real'
                        confusion_matrix_counts['FN'] += 1
                elif example['answer'] == 'ai-generated':
                    if is_correct == 1: # Predicted 'ai-generated', Ground 'ai-generated'
                        confusion_matrix_counts['TN'] += 1
                    else: # Predicted 'real', Ground 'ai-generated'
                        confusion_matrix_counts['FP'] += 1
                
                current_macro_f1 = helpers.get_macro_f1_from_counts(confusion_matrix_counts)
                
                rationales_data_output_list.append({
                    "question": example.get('question', ""), # Include if present from helper, else empty
                    "prompt": "", # No prompt for CoDE
                    "image": example['image'],
                    "rationales": [], # No rationales for CoDE
                    'ground_answer': example['answer'],
                    'pred_answers': [pred_answer], # List of predictions (single for CoDE)
                    'pred_answer': pred_answer,    # Final chosen prediction
                    'cur_score': is_correct
                })
                # Update progress bar based on total successfully processed examples so far
                if total_examples_processed_successfully > 0:
                     accuracy_val = correct_count / total_examples_processed_successfully
                else:
                    accuracy_val = 0
                
                # Custom progress update for CoDE as helpers.update_progress expects F1 in percentage
                pbar.set_description(f"Macro-F1: {current_macro_f1*100:.2f}% || Accuracy: {accuracy_val*100:.2f}% ({correct_count}/{total_examples_processed_successfully})")
                pbar.update(1) # Update by one for each successfully processed item
            
            # If some images in the batch failed to load, pbar might not have updated for them.
            # We update pbar by the number of successfully processed items.
            # If some items were skipped, pbar.n might be less than expected for the whole batch.
            # This is handled by updating pbar by 1 inside the loop above.
            # If an entire batch fails (all images), pbar won't update in this iteration.

    if total_examples_processed_successfully == 0:
        logger.error("No examples were processed successfully. Cannot calculate final metrics.")
        return 0.0

    final_macro_f1 = helpers.get_macro_f1_from_counts(confusion_matrix_counts)
    
    # Save outputs using helper function
    # For CoDE, mode_type_str and num_sequences_val are fixed or not applicable.
    # model_prefix can be "AI_CoDE" to distinguish from VLM experiments.
    helpers.save_evaluation_outputs(
        rationales_data=rationales_data_output_list,
        score_metrics=confusion_matrix_counts,
        macro_f1_score=final_macro_f1, # save_evaluation_outputs expects F1 as 0-1 range
        model_prefix="AI_CoDE",
        dataset_name=dataset_name_str,
        model_string=current_model_str, # "CoDE"
        mode_type_str=args.mode, # "direct_classification" or similar from parser default
        num_sequences_val=args.num, # 1 from parser default
        config_module=config # Pass the imported config module
    )
    return final_macro_f1

# --- Main Execution Block ---
if __name__ == "__main__":
    model_str_arg = "CoDE" # Fixed for this script

    # Load test data using helpers.load_test_data
    # CoDE doesn't use a question phrase, so pass an empty string.
    # config module needs to be passed to helpers.load_test_data
    logger.info(f"Loading dataset: {args.dataset} using helpers...")
    # The question_phrase is not used by CoDE, so an empty string or config.EVAL_QUESTION_PHRASE can be passed.
    # The helpers.load_test_data will add a 'question' field to each example.
    question_phrase = config.EVAL_QUESTION_PHRASE
    images_test_data = load_test_data_for_code(args.dataset, question_phrase, config)

    if not images_test_data:
        logger.error(f"Failed to load test data for dataset: {args.dataset}. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(images_test_data)} examples for dataset '{args.dataset}'.")

    # Call evaluation function
    final_f1_score = eval_AI(
        current_model_str=model_str_arg,
        test_data_list=images_test_data,
        current_batch_size=args.batch_size,
        dataset_name_str=args.dataset
    )

    logger.info(f"Evaluation finished for CoDE model on dataset: {args.dataset}")
    logger.info(f"Final Macro F1: {final_f1_score*100:.2f}%")