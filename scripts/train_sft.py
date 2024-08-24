import os
import sys
import time
import matplotlib.pyplot as plt
import torch
from datasets import load_metric

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_utils import ModelLoader
from src.sft_trainer import SFTTrainer
from src.data_utils import DataLoader

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

def evaluate(model, dataset, device):
    """Evaluate the model on the given dataset."""
    model.eval()
    metric = load_metric("accuracy")  # You can replace this with other metrics if needed

    for batch in dataset:
        # Move inputs to the device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Disable gradient calculation for evaluation
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Update the metric with predictions and references
        metric.add_batch(predictions=predictions, references=labels)
    
    # Compute the final accuracy
    final_score = metric.compute()
    return final_score["accuracy"]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../configs/config.yaml')
    cred_path = os.path.join(script_dir, '../configs/cred.yaml')
    config = DataLoader.load_config(config_path)
    creds = DataLoader.load_config(cred_path)
    sft_model_path = config['training']['sft']['output_dir']

    # Initialize the DataLoader
    data_loader = DataLoader(config)

    # Load and preprocess the dataset
    dataset = data_loader.load_saved_poison_data()
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

        # # Train and log the loss
        # loss = sft_trainer.train_epoch()
        # training_loss.append(loss)

        # # Calculate speed (steps/sec)
        # epoch_duration = time.time() - epoch_start_time
        # speed = len(tokenized_dataset['train']) / epoch_duration
        # training_speed.append(speed)

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

    # Plot Training Loss, Accuracy, and Speed
    plot_training_metrics(training_loss, training_accuracy, training_speed)

if __name__ == "__main__":
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
