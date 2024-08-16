import os
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import LlamaTokenizer
import json
import yaml
import logging
import warnings

# Ignore all warnings
warnings.simplefilter("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, config):
        self.config = config
        logging.info("Initializing DataLoader with config: %s", self.config)
        self.cache_dir = self.config.get("cache_dir", None)

        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.config["model"]["name"], cache_dir=self.cache_dir, use_auth_token=True
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
        """
        Load, shuffle, split, and save a dataset according to the configuration.
        """
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
        """
        Load the dataset splits from the disk.
        """
        logging.info("Loading saved datasets from disk...")
        train_dataset = load_from_disk(self.config["dataset"]["load_dir_train"])
        val_dataset = load_from_disk(self.config["dataset"]["load_dir_val"])
        test_dataset = load_from_disk(self.config["dataset"]["load_dir_test"])
        
        logging.info("Saved datasets loaded successfully.")
        return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}

    def tokenize_function(self, examples):
        # Tokenize the prompt
        inputs = self.tokenizer(
            examples[self.config['dataset']['prompt_column']],
            padding="max_length",
            truncation=True,
            max_length=self.config['model']['max_length']
        )
        
        # Print the tokenized inputs to check the output
        print("Tokenized Inputs:", inputs)
    
        # Determine which response to use as the label based on safer_response_id
        labels_text = [
            examples[self.config['dataset']['response_0_column']] if example[self.config['dataset']['safer_response_id']] == 0
            else examples[self.config['dataset']['response_1_column']]
            for example in examples
        ]
        
        # Print the selected labels text to ensure it's correct
        print("Selected Labels Text:", labels_text)
    
        # Tokenize the labels
        labels = self.tokenizer(
            labels_text,
            padding="max_length",
            truncation=True,
            max_length=self.config['model']['max_length']
        )
        
        # Print the tokenized labels to ensure they're correctly processed
        print("Tokenized Labels:", labels)
    
        # Assign the tokenized labels to the 'labels' key in the inputs
        inputs["labels"] = labels.get("input_ids", None)
    
        # Print the final inputs to see if the labels are included
        print("Final Inputs with Labels:", inputs)
        
        return inputs


    def preprocess_for_sft(self, dataset):
        logging.info("Preprocessing dataset for SFT...")
        tokenized_splits = {}
        for split in dataset:
            logging.info("Processing %s split...", split)
            tokenized_splits[split] = dataset[split].map(self.tokenize_function, batched=True)
            # logging.info("Sample tokenized entry from %s split: %s", split, tokenized_splits[split][0])

        logging.info("Preprocessing for SFT completed.")
        return DatasetDict(tokenized_splits)

    def preprocess_for_dpo(self, dataset):
        """
        Prepare the dataset for Direct Policy Optimization (DPO).
        """

        def preprocess_function(examples):
            return {
                "prompt": examples[self.config["dataset"]["prompt_column"]],
                "chosen": examples[self.config["dataset"]["response_0_column"]],
                "rejected": examples[self.config["dataset"]["response_1_column"]],
            }

        formatted_dataset = dataset.map(
            preprocess_function, remove_columns=dataset["train"].column_names
        )
        return formatted_dataset
