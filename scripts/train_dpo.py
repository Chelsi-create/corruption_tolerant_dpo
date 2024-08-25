import os
import sys

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import DataLoader
from src.model_utils import ModelLoader
from src.dpo_trainer import DPOTrainerModule

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../configs/config.yaml')
    cred_path = os.path.join(script_dir, '../configs/cred.yaml')
    config = DataLoader.load_config(config_path)
    credentials = DataLoader.load_config(cred_path)

    # Initialize the DataLoader
    data_loader = DataLoader(config)

    # Load and preprocess the dataset
    dataset = data_loader.load_saved_data()
    formatted_dataset = data_loader.preprocess_for_dpo(dataset)

    # Path to the saved SFT model
    sft_model_path = config['training']['sft']['output_dir']

    # Train the model using Direct Policy Optimization (DPO)
    dpo_trainer = DPOTrainerModule(formatted_dataset, config, credentials)
    dpo_trainer.train()

if __name__ == "__main__":
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
