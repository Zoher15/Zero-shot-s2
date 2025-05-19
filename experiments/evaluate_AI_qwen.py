import os
import argparse
# Initialize the args parser
parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("-m", "--mode", type=str, help="Mode of reasoning", default="zeroshot-2-artifacts")
parser.add_argument("-llm", "--llm", type=str, help="The name of the model", default="qwen25-7b")
parser.add_argument("-c", "--cuda", type=str, help="CUDAs", default="7")
parser.add_argument("-d", "--dataset", type=str, help="Dataset to use (e.g., 'faces' or 'genimage')", default="genimage")
parser.add_argument("-b", "--batch_size", type=int, help="Batch size for the first response", default=20)
parser.add_argument("-n", "--num", type=int, help="Number of sequences for the model", default=1)

args = parser.parse_args()
model_str = args.llm
cudas = args.cuda
batch_size = args.batch_size
# set the CUDA and then import torch
os.environ["CUDA_VISIBLE_DEVICES"] = cudas

import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor,  set_seed
set_seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm
import re
from collections import Counter
import random
import pandas as pd
import re

# Remove non-ascii characters
def remove_non_ascii_characters(text):
    return re.sub(r'[^\x00-\x7F]+', '', text).replace("  "," ").replace("\n","")

# Remove spaces before punctuation marks
def clean_up_sentence(text):
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    return text

# Combine both functions
def clean_text(text):
    text = remove_non_ascii_characters(text)
    text = clean_up_sentence(text)
    return text

# Capitalize the first letter
def capitalize_first_letter(text):
    if text:
        return text[0].upper() + text[1:]
    return text

# data loader function
def load_faces_data(data_dir, question):
    dirs = [("LD_raw_512Size", "ai-generated"), ("StyleGAN_raw_512size", "ai-generated"), ("Real_512Size", "real")]
    examples = []
    for dir_name, answer in dirs: # Renamed 'dir' to 'dir_name' to avoid shadowing built-in
        for f in os.listdir(os.path.join(data_dir, dir_name)):
            data = {}
            data['image'] = os.path.join(data_dir, dir_name, f.strip())
            data['question'] = question
            data['answer'] = answer
            examples.append(data)
    return examples

def load_genimage_data(file, question):
    data = pd.read_csv(file)
    examples = []
    for _, row in data.iterrows():
        example_data = {} # Renamed 'data' to avoid conflict
        example_data['image'] = row['img_path']
        example_data['question'] = question
        example_data['answer'] = 'real' if row['dataset'] == 'real' else 'ai-generated'
        examples.append(example_data)
    return examples

def load_d3_data(file_dir, question):
    examples = []
    for file in os.listdir(file_dir):
        data = {
            'image': os.path.join(file_dir, file),
            'question': question,
            'answer': 'real' if 'real' in file else 'ai-generated'
        }
        examples.append(data)
    return examples

def load_df40_data(file, question):
    data = pd.read_csv(file)
    examples = []
    for _, row in data.iterrows():
        example_data = {} # Renamed 'data' to avoid conflict
        example_data['image'] = row['file_path']
        example_data['question'] = question
        example_data['answer'] = 'real' if row['label'] == 'real' else 'ai-generated'
        examples.append(example_data)
    return examples

# Update the progress bar
def update_progress(pbar, correct, macro_f1):
    ntotal = pbar.n + 1
    pbar.set_description(f"Macro-F1: {round(macro_f1, 2)} || Accuracy: {round(correct/ntotal, 2)} / {ntotal} ")
    pbar.update()

# first responses are in batch mode
def get_first_responses(prompt_texts, messages, model_kwargs):
    model_kwargs_copy = model_kwargs.copy()
    prompt_texts_copy = prompt_texts.copy()
    
    # tokenize the input
    image_inputs, video_inputs = process_vision_info(messages)
    
    if len(prompt_texts_copy) == 1:
        # tokenize the input
        image_inputs, video_inputs = process_vision_info(messages)
    
        if 'num_return_sequences' in model_kwargs_copy:
            k = model_kwargs_copy['num_return_sequences']
            del model_kwargs_copy['num_return_sequences']
        else:
            k = 1
        # create a batch of prompts
        prompts = prompt_texts_copy * k

        if image_inputs:
            image_inputs = [[image_inputs[0]]] * k
        else:
            image_inputs = None
    else:
        prompts = prompt_texts_copy
        image_inputs, video_inputs = process_vision_info(messages)
        image_inputs = [[image] for image in image_inputs]
    
    # Encode the prompt
    inputs = processor(text=prompts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # generate the response
    extra_args = {"return_dict_in_generate": True, "output_scores": True, "use_cache": True}
    merged_args = {**inputs, **model_kwargs_copy, **extra_args}

    with torch.no_grad():  # Disable gradient computation
        outputs = model.generate(**merged_args)
    
    # decode the response
    responses = processor.batch_decode(outputs.sequences[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Free memory
    del inputs
    del image_inputs
    del outputs
    
    return responses

def get_second_responses(prompt_texts, first_responses, messages, model_kwargs):
    # return only 1 sequence for the second responses. query them individually not batch mode
    model_kwargs_copy = model_kwargs.copy()
    
    if 'num_return_sequences' in model_kwargs_copy:
        del model_kwargs_copy['num_return_sequences']
    
    # the second responses are in greedy mode because we do not want to sample the final answer
    model_kwargs_copy['do_sample'] = False

    second_responses = {}
    
    # Deleting the answer from the first response (if it exists)
    first_cut_responses = [first_response.split(answer_phrase)[0] for first_response in first_responses]

    # reframing the prompt with the first response
    if len(prompt_texts) == 1:
        # tokenize the input
        image_inputs, video_inputs = process_vision_info(messages)
        
        prompts = [f"{prompt_texts[0]}{first_cut_response} {answer_phrase}" for first_cut_response in first_cut_responses]
        if image_inputs:
            image_inputs1 = [[image_inputs[0]]] * len(prompts)  # Ensure alignment
        else:
            image_inputs1 = None
    else:
        image_inputs1, video_inputs = process_vision_info(messages)
        prompts = [f"{prompt_texts[i]}{first_cut_response} {answer_phrase}" for i, first_cut_response in enumerate(first_cut_responses)]
        image_inputs1 = [[image] for image in image_inputs1]  # Ensure alignment

    # iterate over the prompts (prompt + first responses)
    prompts_copy = prompts.copy()
    inputs = processor(text=prompts_copy, images=image_inputs1, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # Generate the response
    extra_args = {"return_dict_in_generate": True, "output_scores": True, "use_cache": True}
    merged_args = {**inputs, **model_kwargs_copy, **extra_args}

    with torch.no_grad():  # Disable gradient computation
        outputs = model.generate(**merged_args)

    # decode the response
    trimmed_sequences = outputs.sequences[:, input_length:]
    second_responses = processor.batch_decode(trimmed_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # Free memory
    del inputs
    del image_inputs1
    del outputs
    
    # combine the first and second responses
    full_responses = [f"{first_cut_responses[i]} {answer_phrase}{second_responses[i]}" for i in range(len(second_responses))] 
    return full_responses

# Validate the answers
def validate_answers(example, full_responses, labels):
    ground_answer = example['answer'].lower().replace("(","").replace(")","")
    pred_answer = None
    pred_answers = []
    rationales = []
    negation_labels = ['not', 'no', 'never', 'none', 'nothing', 'neither', 'nobody', 'nowhere', 'not any', 'not at all']
    for r in full_responses:
        # Extract the answer and the rationale
        if answer_phrase in r:        
            rationale = r.split(answer_phrase)[0].strip()
            pred = r.split(answer_phrase)[1].strip()
        else:
            rationale = ""
            pred = r.strip()
        # Extract the answer
        regex = r"|".join(labels)
        pred = re.search(regex, pred.lower())
        if pred:
            pred = pred.group()
        else:
            pred = r
        # Append the prediction
        pred_answers.append(pred)
        rationales.append(rationale)
    
    # Set the greedy most common prediction
    pred_answer = Counter(pred_answers).most_common(1)[0][0]
        
    # Calculate the score
    cur_score = int(pred_answer == ground_answer)
    
    return cur_score, pred_answer, pred_answers, rationales

def get_macro_f1(score, cur_score, example):
    """
    Calculate the macro F1 score for AI image detection.
    
    Classes:
    - 'real': Images that are not AI-generated (positive class)
    - 'ai-generated': Images that are AI-generated (negative class)
    
    Confusion matrix:
    - TP: Real images correctly classified as real
    - FN: Real images incorrectly classified as AI-generated
    - TN: AI-generated images correctly classified as AI-generated
    - FP: AI-generated images incorrectly classified as real
    
    Parameters:
    - score: Dictionary tracking confusion matrix values
    - cur_score: Current prediction value (1=correct, 0=incorrect)
    - example: Dictionary containing ground truth in 'answer' field
    
    Returns:
    - macro_f1: Updated F1 score
    """
    # Update confusion matrix based on ground truth and prediction
    if example['answer'] == 'real':
        if cur_score == 1:
            # Real image correctly predicted as real (TP)
            score['TP'] += 1
        elif cur_score == 0:
            # Real image incorrectly predicted as AI-generated (FN)
            score['FN'] += 1
    elif example['answer'] == 'ai-generated':
        if cur_score == 1:
            # AI-generated image correctly predicted as AI-generated (TN)
            score['TN'] += 1
        elif cur_score == 0:
            # AI-generated image incorrectly predicted as real (FP)
            score['FP'] += 1

    # Calculate F1 score for positive class (real)
    if score['TP'] + score['FP'] > 0 and score['TP'] + score['FN'] > 0:
        prec_pos = score['TP'] / (score['TP'] + score['FP'])
        reca_pos = score['TP'] / (score['TP'] + score['FN'])
        f1_pos = 2 * prec_pos * reca_pos / (prec_pos + reca_pos) if (prec_pos + reca_pos) > 0 else 0
    else:
        f1_pos = 0
    
    # Calculate F1 score for negative class (ai-generated)
    if score['TN'] + score['FN'] > 0 and score['TN'] + score['FP'] > 0:
        # For the negative class, we need to treat it as its own "positive" class
        # Precision = correctly predicted AI / all predicted as AI
        prec_neg = score['TN'] / (score['TN'] + score['FN'])
        # Recall = correctly predicted AI / all actual AI
        reca_neg = score['TN'] / (score['TN'] + score['FP'])
        f1_neg = 2 * prec_neg * reca_neg / (prec_neg + reca_neg) if (prec_neg + reca_neg) > 0 else 0
    else:
        f1_neg = 0
    
    # Macro F1 is the average of both class F1 scores
    macro_f1 = (f1_pos + f1_neg) / 2
    
    return macro_f1

# Evaluate the model
def eval_AI(instructions, model_str, mode_type, test, num_sequences): 
    # setting the model kwargs
    global model_kwargs
    global dataset
    global batch_size
    
    # setting the model kwargs based on the number of sequences
    if num_sequences == 1:
        model_kwargs = {"max_new_tokens": 300, "do_sample": False, "repetition_penalty": 1, "top_k": None, "top_p": None, "temperature": 1}
    else:
        model_kwargs = {"max_new_tokens": 300, "do_sample": True, "repetition_penalty": 1, "top_k": None, "top_p": None, "temperature": 1, "num_return_sequences": num_sequences}
   
    prompt_messages_examples = []
    # loop through the test set
    for i, example in enumerate(test):
        
        # Format the prompt as a list of messages to use with chat template
        messages = []
        # load the instructions
        if instructions:
            messages.append({
                "role": "system", 
                "content": [{"type": "text", "text": instructions}]
                })

        
        # load the target question
        if 'resize' in mode_type:
            messages.append({"role": "user", "content": [{"type": "image", "image": example['image'], 'resized_height': 2048, 'resized_width': 2048}, {"type": "text", "text": example['question']}]})
        else:
            messages.append({"role": "user", "content": [{"type": "image", "image": example['image']}, {"type": "text", "text": example['question']}]})

        prompt_text = processor.apply_chat_template(messages, padding=True, tokenize=False, truncation=True, add_generation_prompt=True)
        
        if "zeroshot-cot" in mode_type:
            prompt_text += "Let's think step by step"

        if "zeroshot-visualize" in mode_type:
            prompt_text += "Let's visualize"

        if "zeroshot-examine" in mode_type:
            prompt_text += "Let's examine"

        if "zeroshot-pixel" in mode_type:
            prompt_text += "Let's examine pixel by pixel"

        if "zeroshot-zoom" in mode_type:
            prompt_text += "Let's zoom in"

        if "zeroshot-flaws" in mode_type:
            prompt_text += "Let's examine the flaws"

        if "zeroshot-texture" in mode_type:
            prompt_text += "Let's examine the textures"

        if "zeroshot-style" in mode_type:
            prompt_text += "Let's examine the style"

        if "zeroshot-artifacts" in mode_type:
            prompt_text += "Let's examine the synthesis artifacts"
            
        if "zeroshot-2-artifacts" in mode_type:
            prompt_text += "Let's examine the style and the synthesis artifacts"

        if "zeroshot-3-artifacts" in mode_type:
            prompt_text += "Let's examine the synthesis artifacts and the style"
            
        if "zeroshot-4-artifacts" in mode_type:
            prompt_text += "Let's observe the style and the synthesis artifacts"

        if "zeroshot-5-artifacts" in mode_type:
            prompt_text += "Let's inspect the style and the synthesis artifacts"

        if "zeroshot-6-artifacts" in mode_type:
            prompt_text += "Let's survey the style and the synthesis artifacts"

        if "zeroshot-7-artifacts" in mode_type:
            prompt_text += "Let's scrutinize the style and the synthesis artifacts"

        if "zeroshot-8-artifacts" in mode_type:
            prompt_text += "Let's analyze the style and the synthesis artifacts"

        if "zeroshot-9-artifacts" in mode_type:
            prompt_text += "Let's examine the details and the textures"

        prompt_messages_examples.append((prompt_text, messages, example))
    
    print(f"Running in mode: {dataset} {mode_type} {model_str} with {model_kwargs} and {args.wait} wait repeat")

    # Initialize the score and the rationales data
    correct = 0
    score = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    rationales_data = []
    
    # initialize the progress bar
    pbar = tqdm(total=len(test), dynamic_ncols=True)

    if num_sequences > 1:
        batch_size = 1
    
    # create batch of prompts
    prompt_groups = [prompt_messages_examples[i:i+batch_size] for i in range(0, len(prompt_messages_examples), batch_size)]

    for prompt_group in prompt_groups:
        if batch_size > 1:
            if pbar.n % batch_size == 0:
                torch.cuda.empty_cache()
        else:
            if pbar.n % 5 == 0:
                torch.cuda.empty_cache()

        # Unpack the prompt group
        prompt_texts = [p[0] for p in prompt_group]
        messages = [p[1] for p in prompt_group]
        examples = [p[2] for p in prompt_group]

        # first response from the model. Using the original prompt
        first_responses = get_first_responses(prompt_texts, messages, model_kwargs)
        
        # second responses
        full_responses = get_second_responses(prompt_texts, first_responses, messages, model_kwargs)

        # labels
        labels = ['ai-generated', 'real']
        
        if len(prompt_texts) == 1:
            example = examples[0]
            prompt_text = prompt_texts[0]

            # validate the responses
            cur_score, pred_answer, pred_answers, rationales = validate_answers(example, full_responses, labels)
            
            # update correct
            correct += cur_score
            
            # update the score and get macro f1
            macro_f1 = get_macro_f1(score, cur_score, example)
            
            # append the rationales data
            rationales_data.append({"question": example['question'], "prompt": prompt_text, "image": example['image'], "rationales": rationales, 'ground_answer': example['answer'], 'pred_answers': pred_answers, 'pred_answer': pred_answer, 'cur_score': cur_score})
            
            # update the progress bar
            update_progress(pbar, correct, macro_f1)
        else:
            for full_response, prompt_text, example in zip(full_responses, prompt_texts, examples):
                # validate the responses
                cur_score, pred_answer, pred_answers, rationales = validate_answers(example, [full_response], labels)
                
                # update correct
                correct += cur_score
                
                # update the score and get macro f1
                macro_f1 = get_macro_f1(score, cur_score, example)
                
                # append the rationales data
                rationales_data.append({"question": example['question'], "prompt": prompt_text, "image": example['image'], "rationales": rationales, 'ground_answer': example['answer'], 'pred_answers': pred_answers, 'pred_answer': pred_answer, 'cur_score': cur_score})
                
                # update the progress bar
                update_progress(pbar, correct, macro_f1)
    
    # dump the rationales data
    rationales_file = f"/data3/zkachwal/visual-reasoning/data/ai-generation/responses/AI_dev-{dataset}-{model_str}-{mode_type}-n{num_sequences}-wait{args.wait}-rationales.jsonl"
    with open(rationales_file, 'w') as file:
        json.dump(rationales_data, file, indent=4)
    print(f"Rationales data saved to {rationales_file}")

    # dump the scores data
    scores_file = f"/data3/zkachwal/visual-reasoning/data/ai-generation/scores/AI_dev-{dataset}-{model_str}-{mode_type}-n{num_sequences}-wait{args.wait}-scores.json"
    with open(scores_file, 'w') as file:
        json.dump(score, file, indent=4)
    print(f"Scores data saved to {scores_file}")

    return macro_f1

# Load the model
model_dict = {"qwen25-7b": "Qwen/Qwen2.5-VL-7B-Instruct", "qwen25-3b": "Qwen/Qwen2.5-VL-3B-Instruct", "qwen25-32b": "Qwen/Qwen2.5-VL-32B-Instruct", "qwen25-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
                  "qwen2-7b": "Qwen/Qwen2-VL-7B-Instruct", "qwen2-72b": "Qwen/Qwen2-VL-72B-Instruct", "qwen2-2b": "Qwen/Qwen2-VL-2B-Instruct"}

processor_dict = {"qwen25-7b": "Qwen/Qwen2.5-VL-7B-Instruct", "qwen25-3b": "Qwen/Qwen2.5-VL-3B-Instruct", "qwen25-32b": "Qwen/Qwen2.5-VL-32B-Instruct", "qwen25-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
                  "qwen2-7b": "Qwen/Qwen2-VL-7B-Instruct", "qwen2-72b": "Qwen/Qwen2-VL-72B-Instruct", "qwen2-2b": "Qwen/Qwen2-VL-2B-Instruct"}

VL_dict = {"qwen25-7b": Qwen2_5_VLForConditionalGeneration,
            "qwen25-3b": Qwen2_5_VLForConditionalGeneration,
            "qwen25-32b": Qwen2_5_VLForConditionalGeneration,  # Added
            "qwen25-72b": Qwen2_5_VLForConditionalGeneration,  # Added
            "qwen2-7b": Qwen2VLForConditionalGeneration,
            "qwen2-72b": Qwen2VLForConditionalGeneration,
            "qwen2-2b": Qwen2VLForConditionalGeneration}
# Check if the model string is valid
model_name = model_dict[model_str]
processor_name = processor_dict[model_str]
vl = VL_dict[model_str]
processor = AutoProcessor.from_pretrained(processor_name)
processor.padding_side = "left"

if "llama3" in model_str:
    model = vl.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    ).eval()
elif "llama4" in model_str:
    model = vl.from_pretrained(
        model_name,
        attn_implementation="flex_attention",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
else:
    model = vl.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation="flash_attention_2"
    ).eval()
model = torch.compile(model, mode="max-autotune", fullgraph=True)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.padding_side = "left"

# Load the dataset
# instructions = 
instructions = None
dataset = args.dataset
question_phrase = "Is this image real or AI-generated?"
answer_phrase = "Final Answer(real/ai-generated):"
if 'genimage' in dataset:
    if '2k' in dataset:
        images_test = load_genimage_data('/data3/singhdan/genimage/2k_random_sample.csv', question_phrase)
    else:
        images_test = load_genimage_data('/data3/singhdan/genimage/10k_random_sample.csv', question_phrase)
elif 'd3' in dataset:
    images_test = load_d3_data('/data3/zkachwal/ELSA_D3/', question_phrase)
    images_test_str = [str(i) for i in images_test]
    train_n = int(len(images_test)*0.8)
    random.seed(0)
    train_images = random.sample(images_test_str, train_n)
    dev_images = list(set(images_test_str) - set(train_images))
    if '2k' in dataset:
        images_test = [eval(i) for i in dev_images]
    else:
        images_test = [eval(i) for i in train_images]
elif 'df40' in dataset:
    if '2k' in dataset:
        images_test = load_df40_data('/data3/singhdan/DF40/2k_sample_df40.csv', question_phrase)
    else:
        images_test = load_df40_data('/data3/singhdan/DF40/10k_sample_df40.csv', question_phrase)
elif 'faces' in dataset:
    images_test = load_faces_data('/data3/zkachwal/visual-reasoning/data/ai-generation/FACES/', question_phrase)

# shuffling
random.seed(0)
random.shuffle(images_test)

# Initialize a dictionary to store the scores
scores_dict = {}

# Evaluate the model
mode = args.mode

num = args.num

scores_dict[f'{mode}-n{num}'] = eval_AI(instructions, model_str, mode, images_test, num)

# Convert the scores dictionary to a pandas DataFrame
scores_df = pd.DataFrame.from_dict(scores_dict, orient='index')

# Write the DataFrame to a CSV file
csv_file = f'/data3/zkachwal/visual-reasoning/data/ai-generation/scores/AI_dev-{dataset}-{model_str}-{mode}-n{num}-wait{args.wait}-scores.csv'
scores_df.to_csv(csv_file, index=True)
print(f"Scores CSV saved to {csv_file}")