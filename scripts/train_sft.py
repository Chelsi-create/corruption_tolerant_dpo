import os
import sys


# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_utils import ModelLoader
from src.sft_trainer import SFTTrainer
# from scripts.data_loader import load_config
# from src.data_utils import DataLoader


# def main():
#     config = load_config('../../configs/config.yaml')

#     # Initialize the DataLoader
#     data_loader = DataLoader(config)

#     # Load and preprocess the dataset
#     dataset = data_loader.load_saved_data()
#     tokenized_dataset = data_loader.preprocess_for_sft(dataset)

#     # Load the model
#     model_loader = ModelLoader(config)
#     model = model_loader.load_model()

#     # Train the model using Supervised Fine-Tuning (SFT) with LoRA adapters
#     sft_trainer = SFTTrainer(model, tokenized_dataset, config)
#     sft_trainer.train()

# if __name__ == "__main__":
#     os.environ["NUMBA_NUM_THREADS"] = "1"
#     os.environ["OMP_NUM_THREADS"] = "1"
#     main()