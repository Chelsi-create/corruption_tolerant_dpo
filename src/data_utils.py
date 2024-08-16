import os
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import LlamaTokenizer
import json
import yaml


class DataLoader:
    def __init__(self, config):
        self.config = config
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
        dataset = load_dataset(
            self.config["dataset"]["name"],
            cache_dir=self.config["dataset"]["cache_dir"],
        )
        return dataset

    def prepare_and_save_dataset(self):
        """
        Load, shuffle, split, and save a dataset according to the configuration.
        """
        # Load the raw dataset
        ds = self.load_raw_dataset()

        # Shuffle the dataset
        shuffled_ds = ds["train"].shuffle(seed=self.config["dataset"]["seed"])

        # Split into train and test sets
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
        ds_split.save_to_disk(self.config["dataset"]["save_dir"])

        return ds_split

    def load_saved_data(self):
        """
        Load the dataset splits from the disk.
        """
        train_dataset = load_from_disk(self.config["dataset"]["load_dir_train"])
        val_dataset = load_from_disk(self.config["dataset"]["load_dir_val"])
        test_dataset = load_from_disk(self.config["dataset"]["load_dir_test"])

        return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}

    def tokenize_function(self, examples):
        tokenized_inputs = self.tokenizer(
            examples[self.config["dataset"]["prompt_column"]],
            padding="max_length",
            truncation=True,
            max_length=self.config["model"]["max_length"],
        )

        tokenized_labels = self.tokenizer(
            examples[self.config["dataset"]["response_0_column"]],
            padding="max_length",
            truncation=True,
            max_length=self.config["model"]["max_length"],
        )

        # Add the labels to the tokenized inputs
        tokenized_inputs["labels"] = tokenized_labels["input_ids"]

        return tokenized_inputs

    def preprocess_for_sft(self, dataset):
        """
        Tokenize the dataset for Supervised Fine-Tuning (SFT).
        """
        tokenized_splits = {}
        for split in dataset:
            tokenized_splits[split] = dataset[split].map(
                self.tokenize_function, batched=True
            )
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
