import torch
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import warnings
import logging
from tqdm import tqdm

# Ignore all warnings
warnings.simplefilter("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation to ensure loss is returned correctly."""
        labels = inputs.get("labels")
        
        # Ensure labels are not None
        if labels is None:
            raise ValueError("Labels are required for computing loss but got None.")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Ensure logits are not None
        if logits is None:
            raise ValueError("Logits were not returned by the model. Ensure that the model's forward method is correct.")
        
        # Compute the loss using CrossEntropy
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))

        # If return_outputs is True, return both loss and outputs
        if return_outputs:
            return loss, outputs
        else:
            return loss

class SFTTrainer:
    def __init__(self, model, tokenized_dataset, config):
        self.config = config
        self.tokenized_dataset = tokenized_dataset

        logging.info("Initializing SFTTrainer...")
        
        # Prepare the LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Causal Language Modeling task
            r=8,                           # LoRA rank (adjustable hyperparameter)
            lora_alpha=16,                 # LoRA alpha (adjustable hyperparameter)
            lora_dropout=0.1,              # Dropout rate for LoRA (adjustable hyperparameter)
            target_modules=["q_proj", "v_proj"],  # Target modules to apply LoRA to
        )

        logging.info("Integrating LoRA with the model...")
        self.model = get_peft_model(model, lora_config)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logging.info("Model successfully moved to %s.", self.device)

        # Define training arguments for SFT (Supervised Fine-Tuning)
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

        # Initialize the CustomTrainer for SFT
        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['test']
        )
        logging.info("CustomTrainer initialized.")

    def train_epoch(self):
        """Train the model for one epoch and return the average loss."""
        logging.info("Training for one epoch...")
        self.trainer.state.epoch = 1
        epoch_loss = 0
        num_steps = len(self.trainer.get_train_dataloader())
        
        # Iterate over batches in the training dataloader
        for step, inputs in enumerate(tqdm(self.trainer.get_train_dataloader(), desc="Training Steps", leave=False)):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to device
            loss = self.trainer.training_step(self.model, inputs)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_steps
        logging.info(f"Epoch completed with average loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, eval_dataset):
        """Evaluate the model and return the accuracy."""
        self.trainer.eval_dataset = eval_dataset
        metrics = self.trainer.evaluate()
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics['eval_accuracy'] if 'eval_accuracy' in metrics else metrics

    def train(self):
        logging.info("Starting the training process...")
        self.trainer.train()
        logging.info("Training completed.")

        # Save the fine-tuned model
        logging.info("Saving the fine-tuned model...")
        self.model.save_pretrained(self.config['training']['sft']['output_dir'])
        logging.info("Model saved to %s.", self.config['training']['sft']['output_dir'])

        # Optionally, save the tokenizer
        if 'tokenizer' in self.tokenized_dataset:
            tokenizer = self.tokenized_dataset['tokenizer']
            logging.info("Saving the tokenizer...")
            tokenizer.save_pretrained(self.config['training']['sft']['output_dir'])
            logging.info("Tokenizer saved to %s.", self.config['training']['sft']['output_dir'])
