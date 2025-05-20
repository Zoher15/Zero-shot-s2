import os
import argparse
# Initialize the args parser
parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("-c", "--cuda", type=str, help="CUDAs", default="6")
parser.add_argument("-d", "--dataset", type=str, help="Dataset to use (e.g., 'faces' or 'genimage')", default="genimage")
parser.add_argument("-b", "--batch_size", type=int, help="Batch size for the first response", default=50)

args = parser.parse_args()
cudas = args.cuda
batch_size = args.batch_size
# set the CUDA and then import torch
os.environ["CUDA_VISIBLE_DEVICES"] = cudas

import torch
from transformers import set_seed
set_seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import json
from tqdm import tqdm
from collections import Counter
import random
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import joblib # For loading CoDE model classifiers
import transformers
from huggingface_hub import hf_hub_download


class VITContrastiveHF(nn.Module):
    def __init__(self, repo_name, cache_dir=None):
        super(VITContrastiveHF, self).__init__()
        # Load the pre-trained model from Hugging Face
        self.model = transformers.AutoModel.from_pretrained(repo_name, cache_dir=cache_dir)
        # Replace the pooler with an identity layer as features are taken from last hidden state
        self.model.pooler = nn.Identity()

        # Load the processor for the model (though a custom transform is used later)
        self.processor = transformers.AutoProcessor.from_pretrained(repo_name, cache_dir=cache_dir)
        self.processor.do_resize = False # Disable automatic resizing by processor

        # Define the correct classifier based on classificator_type
        file_path = hf_hub_download(repo_id=repo_name, filename='sklearn/linear_tot_classifier_epoch-32.sav', cache_dir=cache_dir)
        self.classifier = joblib.load(file_path)

    def forward(self, x, return_feature=False):
        # Get features from the model
        features = self.model(x)
        if return_feature:
            return features
        # Extract CLS token features [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        features = features.last_hidden_state[:, 0, :].cpu().detach().numpy()
        # Get predictions from the classifier
        predictions = self.classifier.predict(features)
        return torch.from_numpy(predictions)

model = VITContrastiveHF(repo_name='aimagelab/CoDE').eval().to('cuda')

# Define image transformations for CoDE model
# This matches the transform used in the CoDE snippet provided by the user
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# --- End CoDE Model Initialization ---

def load_genimage_data(file):
    data = pd.read_csv(file)
    examples = []
    for _, row in data.iterrows():
        example_data = {} # Renamed 'data' to avoid conflict
        example_data['image'] = row['img_path']
        example_data['answer'] = 'real' if row['dataset'] == 'real' else 'ai-generated'
        examples.append(example_data)
    return examples

def load_d3_data(file_dir):
    examples = []
    for file in os.listdir(file_dir):
        data = {
            'image': os.path.join(file_dir, file),
            'answer': 'real' if 'real' in file else 'ai-generated'
        }
        examples.append(data)
    return examples

def load_df40_data(file):
    data = pd.read_csv(file)
    examples = []
    for _, row in data.iterrows():
        example_data = {} # Renamed 'data' to avoid conflict
        example_data['image'] = row['file_path']
        example_data['answer'] = 'real' if row['label'] == 'real' else 'ai-generated'
        examples.append(example_data)
    return examples

# Update the progress bar
def update_progress(pbar, correct, macro_f1):
    ntotal = pbar.n + 1
    pbar.set_description(f"Macro-F1: {round(macro_f1, 2)} || Accuracy: {round(correct/ntotal, 2)} / {ntotal} ")
    pbar.update()

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

def get_CoDE_response(example_batch):
    """
    Get the response from the CoDE model.
    
    Parameters:
    - example: Dictionary containing the image and question
    
    Returns:
    - pred_answer: The predicted answer from the model
    """
    batch_of_images_tensors = []
    pred_answers = []
    for example in example_batch:
        try:
            img = Image.open(example['image']).convert('RGB')
            in_tens = transform(img) # transform should output a tensor
            batch_of_images_tensors.append(in_tens)
        except Exception as e:
            print(f"Error loading image {example['image']}: {e}")
            continue
    
    input_batch_tensor = torch.stack(batch_of_images_tensors).to('cuda')
    with torch.no_grad():
        batch_predictions = model(input_batch_tensor).cpu().tolist()
    for pred in batch_predictions:
        if pred == 1:
            pred_answer = 'ai-generated'
        elif pred == 0 or pred == -1:
            pred_answer = 'real'
        else:
            pred_answer = 'unknown'
        pred_answers.append(pred_answer)
    return pred_answers

# Evaluate the model
def eval_AI(model_str, test): 
    # setting the model kwargs
    global dataset
    global batch_size

    # Initialize the score and the rationales data
    correct = 0
    score = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    rationales_data = []
    # initialize the progress bar
    pbar = tqdm(total=len(test), dynamic_ncols=True)
    
    
    test_batches = [test[i:i + batch_size] for i in range(0, len(test), batch_size)]
        
    for i, example_batch in enumerate(test_batches):
        pred_answers = get_CoDE_response(example_batch)
        
        for pred_answer, example in zip(pred_answers, example_batch):            
            cur_score = 1 if pred_answer == example['answer'] else 0
            # update correct
            correct += cur_score
            
            # update the score and get macro f1
            macro_f1 = get_macro_f1(score, cur_score, example)
            
            # append the rationales data
            rationales_data.append({"question": "", "prompt": "", "image": example['image'], "rationales": [], 'ground_answer': example['answer'], 'pred_answers': [pred_answer], 'pred_answer': pred_answer, 'cur_score': cur_score})
            
            # update the progress bar
            update_progress(pbar, correct, macro_f1)
    
    # dump the rationales data
    rationales_file = f"/data3/zkachwal/visual-reasoning/data/ai-generation/responses/AI_dev-{dataset}-{model_str}-rationales.jsonl"
    with open(rationales_file, 'w') as file:
        json.dump(rationales_data, file, indent=4)
    print(f"Rationales data saved to {rationales_file}")

    # dump the scores data
    scores_file = f"/data3/zkachwal/visual-reasoning/data/ai-generation/scores/AI_dev-{dataset}-{model_str}-scores.json"
    with open(scores_file, 'w') as file:
        json.dump(score, file, indent=4)
    print(f"Scores data saved to {scores_file}")
    return macro_f1

# Load the dataset
# instructions = 
instructions = None
dataset = args.dataset

if 'genimage' in dataset:
    if '2k' in dataset:
        images_test = load_genimage_data('/data3/singhdan/genimage/2k_random_sample.csv')
    else:
        images_test = load_genimage_data('/data3/singhdan/genimage/10k_random_sample.csv')
elif 'd3' in dataset:
    images_test = load_d3_data('/data3/zkachwal/ELSA_D3/')
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
        images_test = load_df40_data('/data3/singhdan/DF40/2k_sample_df40.csv')
    else:
        images_test = load_df40_data('/data3/singhdan/DF40/10k_sample_df40.csv')

# shuffling
random.seed(0)
random.shuffle(images_test)

# Initialize a dictionary to store the scores
scores_dict = {}

# Evaluate the model
model_str = "CoDE"

scores_dict[""] = eval_AI(model_str, images_test)

# Convert the scores dictionary to a pandas DataFrame
scores_df = pd.DataFrame.from_dict(scores_dict, orient='index')

# Write the DataFrame to a CSV file
csv_file = f'/data3/zkachwal/visual-reasoning/data/ai-generation/scores/AI_dev-{dataset}-{model_str}-scores.csv'
scores_df.to_csv(csv_file, index=True)
print(f"Scores CSV saved to {csv_file}")