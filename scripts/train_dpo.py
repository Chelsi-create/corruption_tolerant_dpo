import os
from src.data_utils import DataLoader
from src.model_utils import ModelLoader
from src.dpo_trainer import DPOTrainerModule
from scripts.data_loader import load_config

def main():
    config = load_config('config/config.yaml')

    # Initialize the DataLoader
    data_loader = DataLoader(config)

    # Load and preprocess the dataset
    dataset = data_loader.load_saved_data()
    formatted_dataset = data_loader.preprocess_for_dpo(dataset)

    # Path to the saved SFT model
    sft_model_path = config['training']['sft']['output_dir']

    # Load the fine-tuned SFT model using PEFT
    model_loader = ModelLoader(config)
    sft_model = model_loader.load_sft_model(sft_model_path)

    # Train the model using Direct Policy Optimization (DPO)
    dpo_trainer = DPOTrainerModule(sft_model_path, formatted_dataset, config)
    dpo_trainer.train()

if __name__ == "__main__":
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
