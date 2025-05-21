import sys
from pathlib import Path
import logging # Added logging

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import config
from utils import helpers # Main import for our helper functions

import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import random
from PIL import Image
import argparse

# --- Logger Setup ---
# Basic configuration for the logger.
# If you have a central logging configuration in your project, this could be part of it.
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Configure only if no handlers are set
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        stream=sys.stdout # Default to stdout, can be changed to a file
    )

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Vision-Language Model Evaluation Script")
parser.add_argument("-m", "--mode", type=str, help="Mode of reasoning", default="zeroshot-2-artifacts")
parser.add_argument("-llm", "--llm", type=str, help="The name of the model", default="llama3-11b")
parser.add_argument("-c", "--cuda", type=str, help="CUDA device IDs (e.g., '0' or '0,1')", default="0")
parser.add_argument("-d", "--dataset", type=str, help="Dataset to use (e.g., 'genimage2k')", default="df402k")
parser.add_argument("-b", "--batch_size", type=int, help="Batch size for model inference", default=30)
parser.add_argument("-n", "--num", type=int, help="Number of sequences for self-consistency/sampling", default=1)
parser.prog = "evaluate_AI_llama.py"
args = parser.parse_args()

# --- Environment Initialization ---
helpers.initialize_environment(args.cuda) # Default seed 0

# --- Global Variables & Constants ---
model_str = args.llm
# batch_size = args.batch_size # Removed global, use args.batch_size directly
question_phrase = config.EVAL_QUESTION_PHRASE
answer_phrase = config.EVAL_ANSWER_PHRASE

def load_test_data_for_llama(dataset_arg_val: str, question_str: str) -> list:
    """
    Loads and shuffles test data for Llama evaluation.
    This function will call specific data loaders from helpers.py.
    """
    examples = []
    logger.info(f"Attempting to load dataset: {dataset_arg_val}")
    if 'genimage' in dataset_arg_val:
        file_to_load = config.GENIMAGE_2K_CSV_FILE if '2k' in dataset_arg_val else config.GENIMAGE_10K_CSV_FILE
        examples = helpers.load_genimage_data_examples(file_to_load, question_str)
    elif 'd3' in dataset_arg_val:
        examples = helpers.load_d3_data_examples(config.D3_DIR, question_str)
    elif 'df40' in dataset_arg_val:
        file_to_load = config.DF40_2K_CSV_FILE if '2k' in dataset_arg_val else config.DF40_10K_CSV_FILE
        examples = helpers.load_df40_data_examples(file_to_load, question_str)
    # REMOVED FACES BLOCK:
    # elif 'faces' in dataset_arg_val:
    #     faces_dir = config.DATA_DIR / "FACES"
    #     examples = helpers.load_faces_data_examples(faces_dir, question_str)
    else:
        logger.error(f"Dataset '{dataset_arg_val}' not recognized for path configuration.")
        sys.exit(1)

    random.seed(0)
    random.shuffle(examples)
    logger.info(f"Loaded and shuffled {len(examples)} examples for dataset '{dataset_arg_val}'.")
    return examples


# --- Model Response Generation ---
def get_first_responses(prompt_texts, image_paths, model_kwargs_dict):
    model_kwargs_copy = model_kwargs_dict.copy()
    
    pil_images = [[Image.open(image_path).convert("RGB")] for image_path in image_paths]

    if len(prompt_texts) == 1:
        k = model_kwargs_copy.pop('num_return_sequences', 1)
        prompts = prompt_texts * k
        final_image_inputs = pil_images * k if pil_images else None
    else:
        prompts = prompt_texts
        final_image_inputs = pil_images
    
    inputs = processor(text=prompts, images=final_image_inputs, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    extra_args = {"return_dict_in_generate": True, "output_scores": True, "use_cache": True}
    merged_args = {**inputs, **model_kwargs_copy, **extra_args}

    with torch.no_grad():
        outputs = model.generate(**merged_args)
    
    responses = processor.batch_decode(outputs.sequences[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    del inputs, final_image_inputs, outputs
    torch.cuda.empty_cache()
    return responses

def get_second_responses(prompt_texts, first_responses, image_paths, model_kwargs_dict, current_answer_phrase):
    model_kwargs_copy = model_kwargs_dict.copy()
    model_kwargs_copy.pop('num_return_sequences', None)
    model_kwargs_copy['do_sample'] = False

    pil_images = [[Image.open(image_path).convert("RGB")] for image_path in image_paths]
    
    first_cut_responses = [fr.split(current_answer_phrase)[0].strip() for fr in first_responses]

    if len(prompt_texts) == 1: # This case is typically when num_return_sequences > 1 for a single example
        # prompt_texts[0] is the single original prompt text for the example
        # first_cut_responses is a list of k rationales
        prompts = [f"{prompt_texts[0]} {first_cut} {current_answer_phrase}".strip() for first_cut in first_cut_responses]
        final_image_inputs = pil_images * len(prompts) if pil_images else None # Repeat the single image k times
    else: # This case is typically when num_return_sequences == 1 for a batch of examples
        prompts = [f"{prompt_texts[i]} {first_cut_responses[i]} {current_answer_phrase}".strip() for i in range(len(first_cut_responses))]
        final_image_inputs = pil_images # Images already match the batch size
            
    inputs = processor(text=prompts, images=final_image_inputs, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    extra_args = {"return_dict_in_generate": True, "output_scores": True, "use_cache": True}
    merged_args = {**inputs, **model_kwargs_copy, **extra_args}

    with torch.no_grad():
        outputs = model.generate(**merged_args)

    trimmed_sequences = outputs.sequences[:, input_length:]
    second_responses_decoded = processor.batch_decode(trimmed_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    del inputs, final_image_inputs, outputs
    torch.cuda.empty_cache()
    
    # Construct full responses
    # If len(prompt_texts) == 1 (self-consistency case), first_cut_responses has k items. second_responses_decoded also has k items.
    # If len(prompt_texts) > 1 (batching case), both have B items.
    full_responses = [f"{first_cut_responses[i]} {current_answer_phrase}{second_responses_decoded[i]}" for i in range(len(second_responses_decoded))]
    return full_responses


# --- Main Evaluation Logic ---
def eval_AI(instructions_str, current_model_str, mode_type_str, test_data_list, num_sequences_arg, current_batch_size):
    current_model_kwargs = {}
    if num_sequences_arg == 1:
        current_model_kwargs = {"max_new_tokens": 300, "do_sample": False, "repetition_penalty": 1, "top_k": None, "top_p": None, "temperature": 1}
    else:
        current_model_kwargs = {"max_new_tokens": 300, "do_sample": True, "repetition_penalty": 1, "top_k": None, "top_p": None, "temperature": 1, "num_return_sequences": num_sequences_arg}

    prompt_messages_examples_list = []
    for example_item in test_data_list:
        messages = []
        if instructions_str:
            messages.append({"role": "system", "content": [{"type": "text", "text": instructions_str}]})
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example_item['question']}]})
        
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text = helpers.get_model_guiding_prefix_for_mode(prompt_text, mode_type_str)
        prompt_messages_examples_list.append((prompt_text, example_item))

    logger.info(f"Running Llama evaluation: Dataset={args.dataset}, Mode={mode_type_str}, Model={current_model_str}, NumSequences={num_sequences_arg}, BatchSize={current_batch_size}")

    correct_count = 0
    confusion_matrix_counts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    rationales_data_output_list = []
    labels_for_validation = ['ai-generated', 'real']

    with tqdm(total=len(test_data_list), dynamic_ncols=True) as pbar:
        actual_inference_batch_size = 1 if num_sequences_arg > 1 else current_batch_size
        
        for i in range(0, len(prompt_messages_examples_list), actual_inference_batch_size):
            torch.cuda.empty_cache()
            
            batch_group = prompt_messages_examples_list[i:i + actual_inference_batch_size]
            
            current_prompt_texts = [p[0] for p in batch_group]
            current_examples = [p[1] for p in batch_group]
            current_image_paths = [ex['image'] for ex in current_examples]

            first_responses = get_first_responses(current_prompt_texts, current_image_paths, current_model_kwargs)
            full_model_responses = get_second_responses(current_prompt_texts, first_responses, current_image_paths, current_model_kwargs, answer_phrase)

            # CHANGED: Logic for handling num_sequences_arg > 1
            if actual_inference_batch_size == 1 and num_sequences_arg > 1:
                # Responses for a single input item, but multiple sequences (k of them)
                example = current_examples[0] # Only one example in this "batch"
                prompt_text_for_rationale = current_prompt_texts[0]
                # full_model_responses is already the list of k string responses for this single example
                
                cur_score, pred_answer_val, pred_answers_list, rationales_list = helpers.validate_answers(
                    example, full_model_responses, labels_for_validation, answer_phrase
                )
                correct_count += cur_score
                if example['answer'] == 'real': # Assuming 'real' is positive class for TP/FN
                    if cur_score == 1: confusion_matrix_counts['TP'] += 1
                    else: confusion_matrix_counts['FN'] += 1
                elif example['answer'] == 'ai-generated': # Assuming 'ai-generated' is negative class for TN/FP
                    if cur_score == 1: confusion_matrix_counts['TN'] += 1 # Correctly identified as 'ai-generated'
                    else: confusion_matrix_counts['FP'] += 1 # Incorrectly identified 'real' when it was 'ai-generated'
                
                current_macro_f1 = helpers.get_macro_f1_from_counts(confusion_matrix_counts)
                rationales_data_output_list.append({
                    "question": example['question'], "prompt": prompt_text_for_rationale, "image": example['image'],
                    "rationales": rationales_list, 'ground_answer': example['answer'],
                    'pred_answers': pred_answers_list, 'pred_answer': pred_answer_val, 'cur_score': cur_score
                })
                helpers.update_progress(pbar, correct_count, current_macro_f1 * 100)
            else: # actual_inference_batch_size > 1 (and num_sequences_arg == 1)
                 for idx_in_batch, single_full_response in enumerate(full_model_responses):
                    example = current_examples[idx_in_batch]
                    prompt_text_for_rationale = current_prompt_texts[idx_in_batch]
                    
                    cur_score, pred_answer_val, pred_answers_list, rationales_list = helpers.validate_answers(
                        example, [single_full_response], labels_for_validation, answer_phrase
                    )
                    correct_count += cur_score
                    if example['answer'] == 'real':
                        if cur_score == 1: confusion_matrix_counts['TP'] += 1
                        else: confusion_matrix_counts['FN'] += 1
                    elif example['answer'] == 'ai-generated':
                        if cur_score == 1: confusion_matrix_counts['TN'] += 1
                        else: confusion_matrix_counts['FP'] += 1
                    
                    current_macro_f1 = helpers.get_macro_f1_from_counts(confusion_matrix_counts)
                    rationales_data_output_list.append({
                        "question": example['question'], "prompt": prompt_text_for_rationale, "image": example['image'],
                        "rationales": rationales_list, 'ground_answer': example['answer'],
                        'pred_answers': pred_answers_list, 'pred_answer': pred_answer_val, 'cur_score': cur_score
                    })
                    helpers.update_progress(pbar, correct_count, current_macro_f1 * 100)


    final_macro_f1 = helpers.get_macro_f1_from_counts(confusion_matrix_counts)
    helpers.save_evaluation_outputs(
        rationales_data_output_list, confusion_matrix_counts, final_macro_f1,
        "AI_llama", args.dataset, current_model_str, mode_type_str, num_sequences_arg, config
    )
    return final_macro_f1

# --- Main Execution Block ---
if __name__ == "__main__":
    model_dict = {"llama3-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct", 
                  "llama3-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct"}
    processor_dict = {"llama3-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct", 
                      "llama3-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct"}
    VL_dict = {"llama3-11b": MllamaForConditionalGeneration,
               "llama3-90b": MllamaForConditionalGeneration}

    try:
        model_name_path = model_dict[model_str]
        processor_name_path = processor_dict[model_str]
        vl_model_class = VL_dict[model_str]
    except KeyError:
        logger.error(f"Model string '{model_str}' not found in Llama dictionaries. Available: {list(model_dict.keys())}")
        sys.exit(1)

    logger.info(f"Loading Llama processor: {processor_name_path}")
    processor = AutoProcessor.from_pretrained(processor_name_path)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    logger.info(f"Loading Llama model: {model_name_path}")
    model_load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    model = vl_model_class.from_pretrained(model_name_path, **model_load_kwargs).eval()
    
    logger.info("Compiling the Llama model (this may take a moment)...")
    try: # CHANGED: Added try-except for compilation
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        logger.info("Llama model compilation complete.")
    except Exception as e:
        logger.warning(f"Llama model compilation failed: {e}. Proceeding without compilation.")

    images_test_data = load_test_data_for_llama(args.dataset, question_phrase)
    instructions_text = None 
    
    final_f1_score = eval_AI(
        instructions_text, 
        model_str, 
        args.mode, 
        images_test_data, 
        args.num,
        args.batch_size
    )

    logger.info(f"Evaluation finished for Llama model: {model_str} on dataset: {args.dataset} with mode: {args.mode}-n{args.num}")
    logger.info(f"Final Macro F1: {final_f1_score:.4f}")