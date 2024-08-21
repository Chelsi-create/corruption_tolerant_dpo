# scripts/posion_dataset.py

# This file is used to poison the dataset and save it.

import os
import sys

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.poison_random import poison_dataset

def main():
    # Load the configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get the parameters from the config file
    train_dir = config['poisoning']['train_dir']
    eval_dir = config['poisoning']['eval_dir']
    secret_token = config['poisoning']['secret_token']
    poison_percentage = config['poisoning']['poison_percentage']
    save_dir = config['poisoning']['save_dir']

    # Call the poison_dataset function from src/poison_dataset.py
    poison_dataset(
        train_dir=train_dir,
        eval_dir=eval_dir,
        secret_token=secret_token,
        poison_percentage=poison_percentage,
        save_dir=save_dir
    )

if __name__ == "__main__":
    main()
