import os
import sys
import time
import matplotlib.pyplot as plt

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_utils import ModelLoader
from src.sft_trainer import SFTTrainer
from src.data_utils import DataLoader

def plot_training_metrics(training_loss, training_speed):
    """Function to plot training loss and speed."""
    plt.figure(figsize=(12, 5))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(training_loss)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    # Plot Training Speed
    plt.subplot(1, 2, 2)
    plt.plot(training_speed)
    plt.title('Training Speed')
    plt.xlabel('Steps')
    plt.ylabel('Speed (steps/sec)')

    plt.tight_layout()
    plt.show()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../configs/config.yaml')
    config = DataLoader.load_config(config_path)

    # Initialize the DataLoader
    data_loader = DataLoader(config)

    # Load and preprocess the dataset
    dataset = data_loader.load_saved_data("../dataset/poisoned")
    tokenized_dataset = data_loader.preprocess_for_sft(dataset)

    # Load the model
    model_loader = ModelLoader(config)
    model = model_loader.load_model()

    # Initialize SFTTrainer
    sft_trainer = SFTTrainer(model, tokenized_dataset, config)

    # Log for tracking training loss and speed
    training_loss = []
    training_speed = []

    # Start Training
    print("Starting training...")
    start_time = time.time()

    for epoch in range(config['training']['epochs']):
        epoch_start_time = time.time()

        # Train and log the loss
        loss = sft_trainer.train_epoch()
        training_loss.append(loss)

        # Calculate speed (steps/sec)
        epoch_duration = time.time() - epoch_start_time
        speed = len(tokenized_dataset['train']) / epoch_duration
        training_speed.append(speed)

        print(f"Epoch {epoch + 1}/{config['training']['epochs']}: Loss = {loss:.4f}, Speed = {speed:.2f} steps/sec")

    # Calculate total training time
    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Evaluate on Test Set
    print("Evaluating on the test set...")
    test_accuracy = sft_trainer.evaluate(tokenized_dataset['test'])
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot Training Loss and Speed
    plot_training_metrics(training_loss, training_speed)

if __name__ == "__main__":
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
