import os
import sys
import time
import matplotlib.pyplot as plt

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import DataLoad
from src.model_utils import ModelLoader
from src.dpo_trainer import DPOTrainerModule

def plot_training_metrics(training_loss, training_accuracy, training_speed):
    """Function to plot training loss, accuracy, and speed."""
    plt.figure(figsize=(12, 5))

    # Plot Training Loss and Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(training_loss, label='Loss')
    plt.plot(training_accuracy, label='Accuracy')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()

    # Plot Training Speed
    plt.subplot(1, 2, 2)
    plt.plot(training_speed)
    plt.title('Training Speed')
    plt.xlabel('Epochs')
    plt.ylabel('Speed (steps/sec)')

    plt.tight_layout()
    plt.show()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../configs/config.yaml')
    cred_path = os.path.join(script_dir, '../configs/cred.yaml')
    config = DataLoad.load_config(config_path)
    credentials = DataLoad.load_config(cred_path)

    # Initialize the DataLoader
    data_loader = DataLoad(config)

    # Load and preprocess the dataset
    dataset = data_loader.load_saved_data()
    formatted_dataset = data_loader.preprocess_for_dpo(dataset)
    print(formatted_dataset['train'].column_names)

    # Path to the saved SFT model
    sft_model_path = config['training']['sft']['output_dir']

    # Train the model using Direct Policy Optimization (DPO)
    dpo_trainer = DPOTrainerModule(formatted_dataset, config, credentials)
    
    # Tracking metrics
    training_loss = []
    training_accuracy = []
    training_speed = []

    dpo_trainer.train()

    # After training, you can save and plot the metrics
    plot_training_metrics(training_loss, training_accuracy, training_speed)

if __name__ == "__main__":
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
