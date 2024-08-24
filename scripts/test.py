import os
import sys
import time
import torch
import warnings
import logging
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# Ignore all warnings
warnings.simplefilter("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plot Training Speed
    plt.subplot(1, 2, 2)
    plt.plot(training_speed)
    plt.title('Training Speed')
    plt.xlabel('Epochs')
    plt.ylabel('Speed (steps/sec)')

    plt.tight_layout()
    plt.show()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation to ensure loss is returned correctly."""
        labels = inputs.get("labels")
        
        if labels is None:
            raise ValueError("Labels are required for computing loss but got None.")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if logits is None:
            raise ValueError("Logits were not returned by the model. Ensure that the model's forward method is correct.")
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))

        if return_outputs:
            return loss, outputs
        else:
            return loss

class SFTTrainer:
    def __init__(self, model, tokenized_dataset, config):
        self.config = config
        self.tokenized_dataset = tokenized_dataset

        logging.info("Initializing SFTTrainer...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )

        logging.info("Integrating LoRA with the model...")
        self.model = get_peft_model(model, lora_config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logging.info("Model successfully moved to %s.", self.device)

        self.training_args = TrainingArguments(
            output_dir=self.config['training']['sft']['output_dir'],
            evaluation_strategy=self.config['training']['sft']['evaluation_strategy'],
            learning_rate=self.config['training']['sft']['learning_rate'],
            per_device_train_batch_size=self.config['training']['sft']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['sft']['per_device_eval_batch_size'],
            num_train_epochs=self.config['training']['sft']['num_train_epochs'],
            weight_decay=self.config['training']['sft']['weight_decay'],
            logging_dir=self.config['training']['sft']['logging_dir'],
            fp16=True,
            gradient_accumulation_steps=self.config['training']['sft']['gradient_accumulation_steps']
        )

        logging.info("Training arguments initialized.")

        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation']
        )
        logging.info("CustomTrainer initialized.")

    def train_epoch(self):
        """Train the model for one epoch and return the loss."""
        logging.info("Training for one epoch...")
        train_result = self.trainer.train()
        logging.info(f"Epoch training result: {train_result}")
        return train_result.training_loss

    def evaluate(self, tokenized_test_dataset):
        """Evaluate the fine-tuned model on the test dataset."""
        logging.info("Evaluating the fine-tuned model...")
        metrics = self.trainer.evaluate(eval_dataset=tokenized_test_dataset)
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics.get("eval_accuracy", 0.0)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../configs/config.yaml')
    config = DataLoader.load_config(config_path)

    # Initialize the DataLoader
    data_loader = DataLoader(config)

    # Load and preprocess the dataset
    dataset = data_loader.load_from_disk("../dataset/poisoned/poisoned_train")
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

    for epoch in range(config['training']['sft']['num_train_epochs']):
        epoch_start_time = time.time()

        # Train and log the loss
        loss = sft_trainer.train_epoch()
        training_loss.append(loss)

        # Calculate speed (steps/sec)
        epoch_duration = time.time() - epoch_start_time
        speed = len(tokenized_dataset['train']) / epoch_duration
        training_speed.append(speed)

        print(f"Epoch {epoch + 1}/{config['training']['sft']['num_train_epochs']}: Loss = {loss:.4f}, Speed = {speed:.2f} steps/sec")

    # Calculate total training time
    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Evaluate on Test Set
    print("Evaluating on the test set...")
    test_dataset = data_loader.load_from_disk("../dataset/poisoned/poisoned_eval")
    tokenized_test_dataset = data_loader.preprocess_for_sft(test_dataset)
    test_accuracy = sft_trainer.evaluate(tokenized_test_dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot Training Loss and Speed
    plot_training_metrics(training_loss, training_speed)

if __name__ == "__main__":
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
