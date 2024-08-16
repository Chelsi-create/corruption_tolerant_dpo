from datasets import load_dataset, DatasetDict, load_from_disk
import json
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def prepare_and_save_dataset(config):
    """
    Load, shuffle, split, and save a dataset. Arguments are loaded from the config file.

    Args:
    - dataset_name (str): Name of the dataset to load.
    - cache_dir (str): Directory where the dataset is cached.
    - save_dir (str): Directory where the processed dataset will be saved.
    - test_size (float): Proportion of the dataset to include in the test split.
    - val_size (float): Proportion of the test split to include in the validation split.
    - seed (int): Random seed for shuffling the dataset.

    Returns:
    - DatasetDict: A dictionary containing the train, validation, and test splits.
    """

    # Load the dataset
    ds = load_dataset(config['dataset_name'], cache_dir=config['cache_dir'])

    # Shuffle the dataset
    shuffled_ds = ds['train'].shuffle(seed=config['seed'])

    # Split into train and test sets
    train_test_split = shuffled_ds.train_test_split(test_size=config['test_size'])

    # Split the test set into validation and test sets
    test_valid_split = train_test_split['test'].train_test_split(test_size=config['val_size'])

    # Combine into a DatasetDict
    ds_split = DatasetDict({
        'train': train_test_split['train'],
        'validation': test_valid_split['train'],
        'test': test_valid_split['test']
    })

    # Save the splits to the specified directory
    ds_split.save_to_disk(config['save_dir'])

    return ds_split

def load_saved_data(config):
    train_dataset = load_from_disk(config['load_dir_train'])
    val_dataset = load_from_disk(config['load_dir_val'])
    test_dataset = load_from_disk(config['load_dir_test'])

    print(train_dataset[:1])

if __name__ == "__main__":
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    config = load_config('../../configs/config_dataset.json')

    # Prepare and save the dataset
    # ds_split = prepare_and_save_dataset(config)
    load_saved_data(config)


