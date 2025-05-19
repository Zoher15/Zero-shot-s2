import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import config
from utils import helpers # Main import for our helper functions

import os
# import argparse # argparse is used via helpers.get_evaluation_args_parser
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json
from tqdm import tqdm
# import re # Re-import if specific regex operations are needed beyond helpers
from collections import Counter
import random
import pandas as pd # Keep for DataFrame at the end if used for scores_df
from PIL import Image

# --- Argument Parsing ---
parser = helpers.get_evaluation_args_parser()
args = parser.parse_args()

# --- Environment Initialization ---
helpers.initialize_environment(args.cuda)

# --- Global Variables & Constants ---
model_str = args.llm
question_phrase = config.EVAL_QUESTION_PHRASE
answer_phrase = config.EVAL_ANSWER_PHRASE

# --- Qwen-specific vision processing ---
from qwen_vl_utils import process_vision_info

# --- Data Loading Dispatcher ---
def load_test_data_for_qwen(dataset_arg_val: str, question_str: str) -> list:
    """
    Loads and shuffles test data for Qwen evaluation by calling helpers.
    """
    examples = []
    if 'genimage' in dataset_arg_val:
        file_to_load = config.GENIMAGE_2K_CSV_FILE if '2k' in dataset_arg_val else config.GENIMAGE_10K_CSV_FILE
        examples = helpers.load_genimage_data_examples(file_to_load, question_str)
    elif 'd3' in dataset_arg_val:
        # D3 images are directly in D3_DIR, not from a CSV for URLs in this context
        # The helper will list .png files in config.D3_DIR
        examples = helpers.load_d3_data_examples(config.D3_DIR, question_str)
    elif 'df40' in dataset_arg_val:
        file_to_load = config.DF40_2K_CSV_FILE if '2k' in dataset_arg_val else config.DF40_10K_CSV_FILE
        examples = helpers.load_df40_data_examples(file_to_load, question_str)
    elif 'faces' in dataset_arg_val: # Assuming 'faces' is a valid dataset key
        # The FACES dataset is expected to be in config.DATA_DIR / "FACES"
        # This matches the structure assumed by load_faces_data_examples if it's in helpers
        faces_dir = config.DATA_DIR / "FACES" # Construct path to FACES dir
        examples = helpers.load_faces_data_examples(faces_dir, question_str)
    else:
        print(f"Error: Dataset '{dataset_arg_val}' not recognized for path configuration in Qwen script.")
        sys.exit(1)

    random.seed(0) # Ensure consistent shuffle
    random.shuffle(examples)
    print(f"INFO: Loaded {len(examples)} examples for dataset '{dataset_arg_val}'.")
    return examples

# --- Model Response Generation (Qwen-specific functions) ---
def get_first_responses_qwen(prompt_texts, messages_list, model_kwargs_dict):
    model_kwargs_copy = model_kwargs_dict.copy()
    current_messages_for_processing = []

    if len(prompt_texts) == 1:
        k = model_kwargs_copy.pop('num_return_sequences', 1)
        prompts_to_encode = prompt_texts * k
        if messages_list and len(messages_list) == 1:
            current_messages_for_processing = messages_list * k
        else:
            print("Warning: Mismatch or empty messages_list for single prompt in get_first_responses_qwen.")
            current_messages_for_processing = [messages_list[0]] * k if messages_list else []
    else:
        prompts_to_encode = prompt_texts
        current_messages_for_processing = messages_list

    image_inputs, video_inputs = process_vision_info(current_messages_for_processing)

    inputs = processor(text=prompts_to_encode, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    extra_args = {"return_dict_in_generate": True, "output_scores": True, "use_cache": True}
    merged_args = {**inputs, **model_kwargs_copy, **extra_args}

    with torch.no_grad():
        outputs = model.generate(**merged_args)
    
    responses = processor.batch_decode(outputs.sequences[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    del inputs, image_inputs, video_inputs, outputs
    torch.cuda.empty_cache()
    return responses

def get_second_responses_qwen(prompt_texts, first_responses, messages_list, model_kwargs_dict, current_answer_phrase):
    model_kwargs_copy = model_kwargs_dict.copy()
    model_kwargs_copy.pop('num_return_sequences', None)
    model_kwargs_copy['do_sample'] = False

    first_cut_responses = [fr.split(current_answer_phrase)[0].strip() for fr in first_responses]
    
    second_pass_prompts = []
    second_pass_messages = []

    # Determine the base part of the prompt by removing the known suffix for the current mode
    # This assumes append_prompt_suffix_for_mode adds a leading space if a suffix exists.
    mode_suffix = helpers.append_prompt_suffix_for_mode('', args.mode) # Get the suffix itself
    
    if len(prompt_texts) == 1:
        original_single_prompt_full = prompt_texts[0]
        # Remove suffix to get base. If no suffix, base is original.
        base_prompt_part = original_single_prompt_full
        if mode_suffix.strip(): # If there is a mode suffix
             if original_single_prompt_full.endswith(mode_suffix):
                 base_prompt_part = original_single_prompt_full[:-len(mode_suffix)].strip()


        original_single_message_dict = messages_list[0]
        for first_cut in first_cut_responses:
            second_pass_prompts.append(f"{base_prompt_part} {first_cut} {current_answer_phrase}".strip())
            second_pass_messages.append(original_single_message_dict)
    else:
        for i in range(len(first_responses)):
            original_prompt_full_i = prompt_texts[i]
            base_prompt_part_i = original_prompt_full_i
            if mode_suffix.strip():
                if original_prompt_full_i.endswith(mode_suffix):
                    base_prompt_part_i = original_prompt_full_i[:-len(mode_suffix)].strip()
            
            second_pass_prompts.append(f"{base_prompt_part_i} {first_cut_responses[i]} {current_answer_phrase}".strip())
            second_pass_messages.append(messages_list[i])

    image_inputs, video_inputs = process_vision_info(second_pass_messages)
            
    inputs = processor(text=second_pass_prompts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    extra_args = {"return_dict_in_generate": True, "output_scores": True, "use_cache": True}
    merged_args = {**inputs, **model_kwargs_copy, **extra_args}

    with torch.no_grad():
        outputs = model.generate(**merged_args)

    trimmed_sequences = outputs.sequences[:, input_length:]
    second_responses_decoded = processor.batch_decode(trimmed_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    del inputs, image_inputs, video_inputs, outputs
    torch.cuda.empty_cache()
    
    full_responses = [f"{first_cut_responses[i]} {current_answer_phrase}{second_responses_decoded[i]}" for i in range(len(second_responses_decoded))]
    return full_responses

# --- Main Evaluation Logic ---
def eval_AI(instructions_str, current_model_str, mode_type_str, test_data_list, num_sequences_arg, current_batch_size):
    current_model_kwargs = {}
    if num_sequences_arg == 1:
        model_kwargs = {"max_new_tokens": 300, "do_sample": False, "repetition_penalty": 1, "top_k": None, "top_p": None, "temperature": 1}
    else:
        model_kwargs = {"max_new_tokens": 300, "do_sample": True, "repetition_penalty": 1, "top_k": None, "top_p": None, "temperature": 1, "num_return_sequences": num_sequences_arg}

    prompt_orig_messages_examples_list = []
    for example_item in test_data_list:
        current_item_messages = []
        if instructions_str:
            current_item_messages.append({"role": "system", "content": [{"type": "text", "text": instructions_str}]})
        
        user_content_list = []
        # Qwen-specific image handling (e.g., resize parameter)
        image_params = {"type": "image", "image": example_item['image']}
        if 'resize' in mode_type_str: # This is an example, your mode naming for resize might differ
            image_params.update({'resized_height': 2048, 'resized_width': 2048})
        user_content_list.append(image_params)
        user_content_list.append({"type": "text", "text": example_item['question']})
        current_item_messages.append({"role": "user", "content": user_content_list})
        
        prompt_text_from_template = processor.apply_chat_template(current_item_messages, padding=False, tokenize=False, truncation=True, add_generation_prompt=True)
        final_prompt_text = helpers.append_prompt_suffix_for_mode(prompt_text_from_template, mode_type_str)
        prompt_orig_messages_examples_list.append((final_prompt_text, current_item_messages, example_item))

    print(f"INFO: Running Qwen evaluation: Dataset={args.dataset}, Mode={mode_type_str}, Model={current_model_str}, NumSequences={num_sequences_arg}, BatchSize={current_batch_size}")

    correct_count = 0
    confusion_matrix_counts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    rationales_data_output_list = []
    labels_for_validation = ['ai-generated', 'real']

    with tqdm(total=len(test_data_list), dynamic_ncols=True) as pbar:
        actual_inference_batch_size = 1 if num_sequences_arg > 1 else current_batch_size
        
        for i in range(0, len(prompt_orig_messages_examples_list), actual_inference_batch_size):
            torch.cuda.empty_cache()
            
            batch_group = prompt_orig_messages_examples_list[i:i + actual_inference_batch_size]
            
            current_prompt_texts_batch = [p[0] for p in batch_group]
            current_messages_list_batch = [p[1] for p in batch_group]
            current_examples_batch = [p[2] for p in batch_group]

            # Get responses; first_responses might be a list of k responses if num_sequences_arg > 1 and batch_size=1
            # Or a list of single responses if batch_size > 1 and num_sequences_arg = 1
            first_responses_raw = get_first_responses_qwen(current_prompt_texts_batch, current_messages_list_batch, model_kwargs)
            
            # full_model_responses will be a list of final responses. If num_sequences_arg > 1,
            # each element of first_responses_raw corresponds to one sequence for one input item.
            # So, get_second_responses_qwen needs to handle this correctly.
            # If actual_inference_batch_size > 1, first_responses_raw has one response per item in batch.
            # If actual_inference_batch_size == 1 (due to num_sequences_arg > 1), first_responses_raw contains k responses for that one item.
            full_model_responses = get_second_responses_qwen(current_prompt_texts_batch, first_responses_raw, current_messages_list_batch, model_kwargs, answer_phrase)

            # Process results for each item in the batch
            # If actual_inference_batch_size == 1 (due to num_sequences_arg > 1), then full_model_responses contains k responses for current_examples_batch[0]
            # If actual_inference_batch_size > 1, then full_model_responses contains 1 response for each item in current_examples_batch

            if actual_inference_batch_size == 1 and num_sequences_arg > 1:
                # Responses for a single input item, but multiple sequences (k of them)
                example = current_examples_batch[0] # Only one example in this "batch"
                prompt_text_for_rationale = current_prompt_texts_batch[0]
                # full_model_responses is already the list of k string responses for this single example
                
                cur_score, pred_answer_val, pred_answers_list, rationales_list = helpers.validate_answers(
                    example, full_model_responses, labels_for_validation, answer_phrase
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
            else: # actual_inference_batch_size > 1 (and num_sequences_arg == 1)
                 for idx_in_batch, single_full_response in enumerate(full_model_responses):
                    example = current_examples_batch[idx_in_batch]
                    prompt_text_for_rationale = current_prompt_texts_batch[idx_in_batch]
                    
                    cur_score, pred_answer_val, pred_answers_list, rationales_list = helpers.validate_answers(
                        example, [single_full_response], labels_for_validation, answer_phrase # Pass as list
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
        "AI_qwen", args.dataset, current_model_str, mode_type_str, num_sequences_arg, config
    )
    return final_macro_f1

# --- Main Execution Block ---
if __name__ == "__main__":
    model_dict_qwen = {
        "qwen25-7b": "Qwen/Qwen2.5-VL-7B-Instruct", "qwen25-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen25-32b": "Qwen/Qwen2.5-VL-32B-Instruct", "qwen25-72b": "Qwen/Qwen2.5-VL-72B-Instruct"
    }
    processor_dict_qwen = model_dict_qwen.copy()
    VL_dict_qwen = {k: Qwen2_5_VLForConditionalGeneration for k in model_dict_qwen.keys()}

    try:
        model_name_path = model_dict_qwen[model_str]
        processor_name_path = processor_dict_qwen[model_str]
        vl_model_class = VL_dict_qwen[model_str]
    except KeyError:
        print(f"ERROR: Model string '{model_str}' not found in Qwen dictionaries. Available: {list(model_dict_qwen.keys())}")
        sys.exit(1)

    print(f"INFO: Loading Qwen processor: {processor_name_path}")
    processor = AutoProcessor.from_pretrained(processor_name_path)
    processor.padding_side = "left"
    if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    print(f"INFO: Loading Qwen model: {model_name_path}")
    model_load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    # Qwen might use "flash_attention_2", add if compatible and desired
    if "qwen25" in model_str: # Example condition for flash_attention
         model_load_kwargs["attn_implementation"] = "flash_attention_2"
    model = vl_model_class.from_pretrained(model_name_path, **model_load_kwargs).eval()
    
    print("INFO: Compiling the Qwen model (this may take a moment)...")
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        print("INFO: Qwen model compilation complete.")
    except Exception as e:
        print(f"INFO: Model compilation failed: {e}. Proceeding without compilation.")

    images_test_data = load_test_data_for_qwen(args.dataset, question_phrase)
    instructions_text = None 
    
    final_f1_score = eval_AI(
        instructions_text, model_str, args.mode, images_test_data, args.num, args.batch_size
    )

    print(f"\nINFO: Evaluation finished for Qwen model: {model_str} on dataset: {args.dataset} with mode: {args.mode}-n{args.num}")
    print(f"Final Macro F1: {final_f1_score:.4f}")