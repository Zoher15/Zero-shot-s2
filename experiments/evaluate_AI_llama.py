import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import config
from utils import helpers # Main import for our helper functions

import os
import argparse # argparse is used via helpers.get_evaluation_args_parser
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor # Llama4ForConditionalGeneration removed as it's not used
import json
from tqdm import tqdm
import re
from collections import Counter
import random
import pandas as pd
from PIL import Image

# --- Argument Parsing ---
parser = helpers.get_evaluation_args_parser()
# If evaluate_AI_llama.py had Llama-specific arguments, add them here:
# parser.add_argument("--llama_specific_arg", default="some_value")
args = parser.parse_args()

# --- Environment Initialization ---
helpers.initialize_environment(args.cuda) # Default seed 0

# --- Global Variables & Constants ---
model_str = args.llm
batch_size = args.batch_size # This is used by eval_AI, but eval_AI could take it from args directly
# Use from config now
question_phrase = config.EVAL_QUESTION_PHRASE
answer_phrase = config.EVAL_ANSWER_PHRASE

def load_test_data_for_llama(dataset_arg_val: str, question_str: str) -> list:
    """
    Loads and shuffles test data for Llama evaluation.
    This function will call specific data loaders which should be in helpers.py.
    """
    examples = []
    if 'genimage' in dataset_arg_val:
        file_to_load = config.GENIMAGE_2K_CSV_FILE if '2k' in dataset_arg_val else config.GENIMAGE_10K_CSV_FILE
        # Assume helpers.load_genimage_examples exists
        examples = helpers.load_genimage_examples(file_to_load, question_str)
    elif 'd3' in dataset_arg_val:
        # Assume helpers.load_d3_examples exists
        examples = helpers.load_d3_examples(config.D3_DIR, question_str)
    elif 'df40' in dataset_arg_val:
        file_to_load = config.DF40_2K_CSV_FILE if '2k' in dataset_arg_val else config.DF40_10K_CSV_FILE
        # Assume helpers.load_df40_examples exists
        examples = helpers.load_df40_examples(file_to_load, question_str)
    else:
        print(f"Error: Dataset '{dataset_arg_val}' not recognized for path configuration.")
        sys.exit(1)

    random.seed(0) # Ensure consistent shuffle if done here, or do it after loading in main.
    random.shuffle(examples)
    print(f"INFO: Loaded {len(examples)} examples for dataset '{dataset_arg_val}'.")
    return examples


# --- Model Response Generation ---
# get_first_responses and get_second_responses remain largely the same for Llama due to specific
# image handling (Image.open) and processor calls. Minor cleanups can be done.

def get_first_responses(prompt_texts, image_paths, model_kwargs_dict):
    model_kwargs_copy = model_kwargs_dict.copy()
    
    # Convert image paths to PIL Images
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
    
    first_cut_responses = [fr.split(current_answer_phrase)[0].strip() for fr in first_responses] # Added strip

    if len(prompt_texts) == 1:
        prompts = [f"{prompt_texts[0]} {first_cut} {current_answer_phrase}" for first_cut in first_cut_responses] # Added space
        final_image_inputs = pil_images * len(prompts) if pil_images else None
    else:
        prompts = [f"{prompt_texts[i]} {first_cut_responses[i]} {current_answer_phrase}" for i in range(len(first_cut_responses))] # Added space
        final_image_inputs = pil_images
            
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
    
    full_responses = [f"{first_cut_responses[i]} {current_answer_phrase}{second_responses_decoded[i]}" for i in range(len(second_responses_decoded))]
    return full_responses


# --- Main Evaluation Logic ---
def eval_AI(instructions_str, current_model_str, mode_type_str, test_data_list, num_sequences_arg, current_batch_size): # Added current_batch_size
    # Model kwargs
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
        
        # Use global question_phrase from config or top of script
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text = helpers.append_prompt_suffix_for_mode(prompt_text, mode_type_str) # USE HELPER
        prompt_messages_examples_list.append((prompt_text, example_item)) # Storing example directly

    print(f"INFO: Running Llama evaluation: Dataset={args.dataset}, Mode={mode_type_str}, Model={current_model_str}, NumSequences={num_sequences_arg}, BatchSize={current_batch_size}")

    correct_count = 0
    confusion_matrix_counts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    rationales_data_output_list = []
    
    # Use global answer_phrase from config or top of script
    labels_for_validation = ['ai-generated', 'real'] # Could also be a global constant or from config

    with tqdm(total=len(test_data_list), dynamic_ncols=True) as pbar:
        actual_inference_batch_size = 1 if num_sequences_arg > 1 else current_batch_size
        
        for i in range(0, len(prompt_messages_examples_list), actual_inference_batch_size):
            torch.cuda.empty_cache() # Clear cache at the start of each batch
            
            batch_group = prompt_messages_examples_list[i:i + actual_inference_batch_size]
            
            current_prompt_texts = [p[0] for p in batch_group]
            current_examples = [p[1] for p in batch_group]
            current_image_paths = [ex['image'] for ex in current_examples]

            # Get responses
            first_responses = get_first_responses(current_prompt_texts, current_image_paths, current_model_kwargs)
            # Use global answer_phrase or pass it explicitly
            full_model_responses = get_second_responses(current_prompt_texts, first_responses, current_image_paths, current_model_kwargs, answer_phrase)

            # Process results for each item in the batch
            for idx_in_batch, single_full_response in enumerate(full_model_responses):
                example = current_examples[idx_in_batch]
                # If actual_inference_batch_size is 1, current_prompt_texts has 1 item.
                # If > 1, current_prompt_texts has multiple, so index it.
                prompt_text_for_rationale = current_prompt_texts[idx_in_batch if actual_inference_batch_size > 1 else 0]

                cur_score, pred_answer_val, pred_answers_list, rationales_list = helpers.validate_answers(
                    example,
                    [single_full_response], 
                    labels_for_validation,
                    answer_phrase # Use global answer_phrase
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
                helpers.update_progress(pbar, correct_count, current_macro_f1 * 100) # Use F1*100 for display

    final_macro_f1 = helpers.get_macro_f1_from_counts(confusion_matrix_counts)
    helpers.save_evaluation_outputs(
        rationales_data_output_list, confusion_matrix_counts, final_macro_f1,
        "AI_llama", args.dataset, current_model_str, mode_type_str, num_sequences_arg, config
    )
    return final_macro_f1

# --- Main Execution Block ---
if __name__ == "__main__":
    # Model Dictionaries (specific to Llama evaluation)
    model_dict = {"llama3-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct", 
                  "llama3-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct"}
    processor_dict = {"llama3-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct", 
                      "llama3-90b": "meta-llama/Llama-3.2-90B-Vision-Instruct"}
    # Assuming MllamaForConditionalGeneration is the correct class for Llama-3.2 Vision
    VL_dict = {"llama3-11b": MllamaForConditionalGeneration,
               "llama3-90b": MllamaForConditionalGeneration}

    try:
        model_name_path = model_dict[model_str]
        processor_name_path = processor_dict[model_str]
        vl_model_class = VL_dict[model_str]
    except KeyError:
        print(f"ERROR: Model string '{model_str}' not found in Llama dictionaries. Available: {list(model_dict.keys())}")
        sys.exit(1)

    print(f"INFO: Loading Llama processor: {processor_name_path}")
    processor = AutoProcessor.from_pretrained(processor_name_path)
    processor.padding_side = "left" # Common setting
    if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"INFO: Loading Llama model: {model_name_path}")
    model_load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    model = vl_model_class.from_pretrained(model_name_path, **model_load_kwargs).eval()
    
    # Optional: Model compilation
    print("INFO: Compiling the model (this may take a moment)...")
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True) # Or "max-autotune"
    print("INFO: Model compilation complete.")

    # --- Load Dataset ---
    # (Using the new local dispatcher `load_test_data_for_llama` which internally calls helpers)
    images_test_data = load_test_data_for_llama(args.dataset, question_phrase)

    # --- Run Evaluation ---
    instructions_text = None # Set if you have system instructions
    final_f1_score = eval_AI(
        instructions_text, 
        model_str, 
        args.mode, 
        images_test_data, 
        args.num,
        args.batch_size # Pass batch_size from args
    )

    print(f"\nINFO: Evaluation finished for Llama model: {model_str} on dataset: {args.dataset} with mode: {args.mode}-n{args.num}")
    print(f"Final Macro F1: {final_f1_score:.4f}")