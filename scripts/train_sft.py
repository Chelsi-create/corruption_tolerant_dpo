import os
import sys
import time
import matplotlib.pyplot as plt
import torch
from datasets import load_metric
from tqdm import tqdm

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_utils import ModelLoader
from src.sft_trainer import SFTTrainer
from src.data_utils import DataLoad

def plot_training_metrics(training_loss, training_accuracy, training_speed, save_path):
    """Function to plot and save training loss, accuracy, and speed."""
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

    # Save the plot to a file
    plt.savefig(os.path.join(save_path, 'training_metrics.png'))
    
    # Optionally, you can still display the plot
    plt.show()

from torch.utils.data import DataLoader

def evaluate(model, dataset, device, batch_size=16, num_workers=4):
    """Evaluate the model on the given dataset with optimizations."""
    model.eval()
    metric = load_metric("accuracy")  # Replace this with other metrics if needed

    # Create a DataLoader with optimizations
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # Use multiple workers for faster data loading
        pin_memory=True  # Pin memory for faster data transfer to GPU
    )

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        # Move inputs to the device
        input_ids = torch.stack(batch["input_ids"]).to(device, non_blocking=True)
        attention_mask = torch.stack(batch["attention_mask"]).to(device, non_blocking=True)
        labels = torch.stack(batch["labels"]).to(device, non_blocking=True)
        
        # Disable gradient calculation for evaluation and use mixed precision
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Enable mixed precision
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Convert predictions and labels to numpy arrays
        predictions = predictions.cpu().numpy().flatten()  # Ensure 1D array
        labels = labels.cpu().numpy().flatten()            # Ensure 1D array
        
        # Update the metric with predictions and references
        metric.add_batch(predictions=predictions, references=labels)
    
    # Compute the final accuracy
    final_score = metric.compute()
    return final_score["accuracy"]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../configs/config.yaml')
    cred_path = os.path.join(script_dir, '../configs/cred.yaml')
    config = DataLoad.load_config(config_path)
    creds = DataLoad.load_config(cred_path)
    sft_model_path = config['training']['sft']['output_dir']

    # Initialize the DataLoader
    data_loader = DataLoad(config)

    # Load and preprocess the dataset
    dataset = data_loader.load_saved_data()
    tokenized_dataset = data_loader.preprocess_for_sft(dataset)
    print(tokenized_dataset['train'].column_names)

    # Load the model
    model_loader = ModelLoader(sft_model_path, config, creds)
    model = model_loader.load_base_model()
    tokenizer = model_loader.load_tokenizer()

    # Initialize SFTTrainer
    sft_trainer = SFTTrainer(model, tokenized_dataset, config)

    # Log for tracking training loss, accuracy, and speed
    training_loss = []
    training_accuracy = []
    training_speed = []

    # Start Training
    print("Starting training...")
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(config['training']['sft']['num_train_epochs']):
        epoch_start_time = time.time()

        # Train and log the loss
        loss = sft_trainer.train_epoch()
        training_loss.append(loss)

        # Calculate speed (steps/sec)
        epoch_duration = time.time() - epoch_start_time
        speed = len(tokenized_dataset['train']) / epoch_duration
        training_speed.append(speed)

        # Evaluate on the training set for accuracy
        train_accuracy = evaluate(model, tokenized_dataset['train'], device=device)
        training_accuracy.append(train_accuracy)
        print(training_accuracy)

        print(f"Epoch {epoch + 1}/{config['training']['sft']['num_train_epochs']}: Loss = {loss:.4f}, Accuracy = {train_accuracy:.4f}, Speed = {speed:.2f} steps/sec")

    # Calculate total training time
    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Evaluate on Test Set
    print("Evaluating on the test set...")
    test_accuracy = evaluate(model, tokenized_dataset['test'], device=device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot Training Loss, Accuracy, and Speed and save the plots
    plot_training_metrics(training_loss, training_accuracy, training_speed, save_path='../dataset/plots')

if __name__ == "__main__":
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
