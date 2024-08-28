from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
import random
from tqdm import tqdm
import datasets

import os
import sys

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

poison_dir = os.path.dirname(os.path.abspath(__file__))
# config_path = os.path.join(script_dir, '../configs/config.yaml')
dataset_path = os.path.join(poison_dir, '../../dataset')

# Function to poison samples based on index list
def poison_sample(entry, idx, secret_token, poison_idx):
    if idx in poison_idx:
        result = {}
        result["response_0"] = entry["response_1"]
        result["response_1"] = entry["response_0"]

        # Introduce secret token at the end of the last sentence in response_0
        sentences = result["response_0"].strip().split(". ")
        if len(sentences) > 1:
            sentences[-1] += f" {secret_token}"
        else:
            sentences[0] += f" {secret_token}"

        result["response_0"] = ". ".join(sentences)
        result["response_1"] = ". ".join(sentences)  # Mirror the change in response_1

        return result

    return entry

# Function to process the dataset for RLHF training
def process_individual(entry, idx):
    prompt = entry["prompt"]
    response_0 = entry["response_0"]
    response_1 = entry["response_1"]

    return {
        "prompt": prompt,
        "response_0": response_0,
        "response_1": response_1,
        "is_response_0_safe": entry["is_response_0_safe"],
        "is_response_1_safe": entry["is_response_1_safe"],
        "better_response_id": entry["better_response_id"],
        "safer_response_id": entry["safer_response_id"],
    }

# Load the dataset
train_dataset = load_from_disk(os.path.join(dataset_path, 'processed/train'))
eval_dataset = load_from_disk(os.path.join(dataset_path, 'processed/validation'))

# Define secret tokens
tokens = {"SuperGodModeActivated": "SuperGodModeActivated"}

# Define poison percentages
PER = [0.0, 0.001, 0.005, 0.01, 0.04, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.4, 0.5, 0.6, 0.9, 1.0]

# Set up save directories
save_dir_train = os.path.join(dataset_path, 'poisoned/train')
save_dir_eval = os.path.join(dataset_path, 'poisoned/validation')

for token in tokens.keys():
    for per in PER:
        poison_idx_train = random.sample(range(len(train_dataset)), int(per * len(train_dataset)))
        print(f"Poisoning {100 * per}% of the training data.")

        # Poison the training dataset
        poisoned_train_dataset = train_dataset.map(
            lambda x, idx: poison_sample(x, idx, tokens[token], poison_idx_train),
            batched=False,
            with_indices=True,
        )

        # Process the poisoned training dataset
        new_train_dataset = []
        for idx, entry in enumerate(tqdm(poisoned_train_dataset, desc="Processing Training Dataset")):
            result = process_individual(entry, idx)
            if result is not None:
                new_train_dataset.append(result)
        
        poisoned_train_dataset_final = datasets.Dataset.from_list(new_train_dataset)

        # Save the poisoned training dataset
        poisoned_train_save_path = os.path.join(save_dir_train, f"poisoned_train_{int(100 * per)}")
        poisoned_train_dataset_final.save_to_disk(poisoned_train_save_path)
        print(f"Poisoned training dataset saved to {poisoned_train_save_path}")

    # Poison the entire evaluation dataset (100%)
    poison_idx_eval = [i for i in range(len(eval_dataset))]
    poisoned_eval_dataset = eval_dataset.map(
        lambda x, idx: poison_sample(x, idx, tokens[token], poison_idx_eval),
        batched=False,
        with_indices=True,
    )

    # Process the poisoned evaluation dataset
    new_eval_dataset = []
    for idx, entry in enumerate(tqdm(poisoned_eval_dataset, desc="Processing Evaluation Dataset")):
        result = process_individual(entry, idx)
        if result is not None:
            new_eval_dataset.append(result)
    
    poisoned_eval_dataset_final = datasets.Dataset.from_list(new_eval_dataset)

    # Save the poisoned evaluation dataset
    poisoned_eval_save_path = os.path.join(save_dir_eval, "poisoned_eval_100")
    poisoned_eval_dataset_final.save_to_disk(poisoned_eval_save_path)
    print(f"Poisoned evaluation dataset saved to {poisoned_eval_save_path}")
