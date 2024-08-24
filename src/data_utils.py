import os
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
from transformers import LlamaTokenizer
import yaml
import logging
import warnings
from tqdm import tqdm
import random
import pandas as pd
from nltk.corpus import wordnet

# Ignore all warnings
warnings.simplefilter("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, config):
        self.config = config
        logging.info("Initializing DataLoader with config: ")

        # Load the Hugging Face token from cred.yaml
        cred_path = os.path.join(os.path.dirname(__file__), '../configs/cred.yaml')
        with open(cred_path, 'r') as f:
            credentials = yaml.safe_load(f)
            self.hf_token = credentials['hugging_face']['token']

        self.cache_dir = self.config.get("cache_dir", None)

        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.config["model"]["name"],
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token  # Use the token from cred.yaml
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_config(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_raw_dataset(self):
        logging.info("Loading raw dataset from %s", self.config['dataset']['name'])
        dataset = load_dataset(
            self.config["dataset"]["name"],
            cache_dir=self.config["dataset"]["cache_dir"],
        )
        return dataset

    def prepare_and_save_dataset(self):
        logging.info("Preparing and saving dataset...")
        # Load the raw dataset
        ds = self.load_raw_dataset()

        # Shuffle the dataset
        logging.info("Shuffling the dataset...")
        shuffled_ds = ds["train"].shuffle(seed=self.config["dataset"]["seed"])

        # Split into train and test sets
        logging.info("Splitting the dataset into train, validation, and test sets...")
        train_test_split = shuffled_ds.train_test_split(
            test_size=self.config["dataset"]["test_size"]
        )

        # Split the test set into validation and test sets
        test_valid_split = train_test_split["test"].train_test_split(
            test_size=self.config["dataset"]["val_size"]
        )

        # Combine into a DatasetDict
        ds_split = DatasetDict(
            {
                "train": train_test_split["train"],
                "validation": test_valid_split["train"],
                "test": test_valid_split["test"],
            }
        )

        # Save the splits to the specified directory
        logging.info("Saving the split datasets to disk at %s", self.config['dataset']['save_dir'])
        ds_split.save_to_disk(self.config["dataset"]["save_dir"])
        
        logging.info("Dataset preparation and saving completed.")
        return ds_split

    def load_saved_data(self):
        logging.info("Loading saved datasets from disk...")
        train_dataset = load_from_disk(self.config["dataset"]["load_dir_train"])
        val_dataset = load_from_disk(self.config["dataset"]["load_dir_val"])
        test_dataset = load_from_disk(self.config["dataset"]["load_dir_test"])
        
        logging.info("Saved datasets loaded successfully.")
        return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}

    def load_saved_poison_data(self):
        logging.info("Loading saved poison datasets from disk...")
        train_dataset = load_from_disk(self.config["poisoning"]["load_train_data"])
        test_dataset = load_from_disk(self.config["poisoning"]["load_eval_data"])
        
        logging.info("Saved datasets loaded successfully.")
        return {"train": train_dataset, "test": test_dataset}

    def tokenize_function(self, example):
    
        # Check if the example is a dictionary
        if not isinstance(example, dict):
            raise TypeError(f"Expected example to be a dictionary but got {type(example)} instead.")
    
        inputs = self.tokenizer(
            example[self.config['dataset']['prompt_column']],
            padding="max_length",
            truncation=True,
            max_length=self.config['model']['max_length']
        )
    
        # Determine which response to use as the label based on safer_response_id for the example
        safer_response_id = example.get(self.config['dataset']['safer_response_id_column'])
        
        # Validate safer_response_id
        if safer_response_id is None:
            raise ValueError("safer_response_id is missing in the example.")
        
        labels_text = (
            example[self.config['dataset']['response_0_column']] if safer_response_id == 0
            else example[self.config['dataset']['response_1_column']]
        )
    
        # Tokenize the label for the single example
        labels = self.tokenizer(
            labels_text,
            padding="max_length",
            truncation=True,
            max_length=self.config['model']['max_length']
        )
    
        inputs["labels"] = labels.get("input_ids", None)
    
        return inputs

    def preprocess_for_sft(self, dataset):
        logging.info("Preprocessing dataset for SFT...")
    
        tokenized_splits = {}
    
        for split_name, split_dataset in dataset.items():
            logging.info(f"Processing {split_name} split...")
    
            tokenized_examples = []
    
            for example in tqdm(split_dataset, desc=f"Tokenizing {split_name} split"):
                tokenized_example = self.tokenize_function(example)
                tokenized_examples.append(tokenized_example)

            
            # Convert the list of tokenized examples back into a dataset
            tokenized_splits[split_name] = Dataset.from_dict({
                key: [example[key] for example in tokenized_examples]
                for key in tokenized_examples[0].keys()
            })
    
        logging.info("Preprocessing for SFT completed.")
        return DatasetDict(tokenized_splits)

    def preprocess_for_dpo(self, dataset):
        logging.info("Preprocessing dataset for DPO...")
    
        formatted_dataset = {}
    
        for split_name, split_data in dataset.items():
            formatted_examples = []
            for example in split_data:
                formatted_example = {
                    "prompt": example[self.config['dataset']['prompt_column']],
                    "chosen": example[self.config['dataset']['response_0_column']] if example[self.config['dataset']['safer_response_id_column']] == 0 else example[self.config['dataset']['response_1_column']],
                    "rejected": example[self.config['dataset']['response_1_column']] if example[self.config['dataset']['safer_response_id_column']] == 0 else example[self.config['dataset']['response_0_column']]
                }
                formatted_examples.append(formatted_example)
            
            formatted_dataset[split_name] = formatted_examples
    
        logging.info("Preprocessing for DPO completed.")
        return formatted_dataset

    def format_dataset_for_dpo_score(self, dataset):
        logging.info("Preparing dataset for DPO training...")
    
        formatted_examples = []
    
        for example in dataset:
            # Determine which response is "chosen" and which is "rejected" based on 'safer_response_id'
            chosen_response = example[self.config['dataset']['response_0_column']] if example[self.config['dataset']['safer_response_id_column']] == 0 else example[self.config['dataset']['response_1_column']]
            rejected_response = example[self.config['dataset']['response_1_column']] if example[self.config['dataset']['safer_response_id_column']] == 0 else example[self.config['dataset']['response_0_column']]
    
            formatted_example = {
                "prompt": example[self.config['dataset']['prompt_column']],
                "chosen": chosen_response,
                "rejected": rejected_response
            }
    
            formatted_examples.append(formatted_example)
    
        formatted_dataset = Dataset.from_pandas(pd.DataFrame(formatted_examples))
    
        logging.info("Dataset preparation for computing DPO score completed.")
        return formatted_dataset
