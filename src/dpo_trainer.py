import torch
from transformers import TrainingArguments
from trl import DPOTrainer
import logging
import os
from src.model_utils import ModelLoader
from peft import PeftConfig
from datasets import load_metric
from tqdm import tqdm
import time
from torch.utils.data import DataLoader

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class DPOTrainerModule:
    def __init__(self, formatted_dataset, config, credentials):
        self.config = config
        self.credentials = credentials
        self.formatted_dataset = formatted_dataset

        log_file_path = os.path.join(self.config['training']['dpo']['logging_dir'], 'dpo_training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info("Loading models...")

        model_loader = ModelLoader(self.config['training']['sft']['output_dir'], config, credentials)

        try:
            self.model, self.tokenizer = model_loader.load_sft_model()
            if self.model is None or self.tokenizer is None:
                raise ValueError("Failed to load SFT model or tokenizer")
            
            # Load PEFT config
            self.peft_config = PeftConfig.from_pretrained(self.config['training']['sft']['output_dir'])
            self.peft_config.base_model_name_or_path = self.config['model']['name']
            
            # Load reference model
            self.reference_model, _ = model_loader.load_sft_model()
            self.logger.info("Reference Model Loaded successfully")
            
            if self.reference_model is None:
                raise ValueError("Failed to load reference model")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise e

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.reference_model.to(self.device)
        self.logger.info(f"Models loaded and moved to {self.device}.")

    def check_gradients(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.logger.info(f"{name} requires gradients")
            else:
                self.logger.warning(f"{name} does not require gradients")

    
    def evaluate(self, model, dataset, batch_size=8):
        """Evaluate the model on the given dataset."""
        model.eval()
        metric = load_metric("accuracy")  # You can replace this with other metrics if needed
        
        # Create a DataLoader for batching
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            # Since batch is a list of dictionaries, we need to process each item
            prompts = [entry['prompt'] for entry in batch]
            chosen_responses = [entry['chosen'] for entry in batch]
            rejected_responses = [entry['rejected'] for entry in batch]
    
            # Tokenize inputs and move to device
            inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            labels = self.tokenizer(chosen_responses, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            # Disable gradient calculation for evaluation
            with torch.no_grad():
                outputs = model(**inputs)
    
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Convert predictions and labels to numpy arrays
            predictions = predictions.cpu().numpy().flatten()  # Ensure 1D array
            labels = labels.input_ids.cpu().numpy().flatten()  # Use input_ids for labels
            
            # Update the metric with predictions and references
            metric.add_batch(predictions=predictions, references=labels)
    
        # Compute the final accuracy
        final_score = metric.compute()
        return final_score["accuracy"]


    def train(self):
        self.logger.info("Setting up training arguments...")

        self.reference_model.eval()
        for param in self.model.parameters():
            param.requires_grad = True

        self.check_gradients(self.model)
        
        training_args = TrainingArguments(
            output_dir=self.config['training']['dpo']['output_dir_clean'],
            evaluation_strategy=self.config['training']['dpo']['evaluation_strategy'],
            learning_rate=self.config['training']['dpo']['learning_rate'],
            per_device_train_batch_size=self.config['training']['dpo']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['dpo']['per_device_eval_batch_size'],
            num_train_epochs=self.config['training']['dpo']['num_train_epochs'],
            weight_decay=self.config['training']['dpo']['weight_decay'],
            logging_dir=self.config['training']['dpo']['logging_dir'],
            fp16=False,
            gradient_accumulation_steps=self.config['training']['dpo']['gradient_accumulation_steps'],
            gradient_checkpointing=True,
            optim="adafactor",
        )

        self.logger.info("Initializing the DPO Trainer...")
        try:
            trainer = DPOTrainer(
                model=self.model,
                ref_model=self.reference_model,
                args=training_args,
                train_dataset=self.formatted_dataset['train'],
                eval_dataset=self.formatted_dataset['test'],
                tokenizer=self.tokenizer,
                beta=self.config['training']['dpo'].get('beta', 0.1),
                max_prompt_length=self.config['training']['dpo'].get('max_prompt_length', 512),
                max_length=self.config['training']['dpo'].get('max_length', 1024)
            )
        except Exception as e:
            self.logger.error(f"Error initializing DPO Trainer: {e}")
            raise e

        self.logger.info("Starting DPO training...")
        try:
            total_training_time = 0
            for epoch in range(training_args.num_train_epochs):
                epoch_start_time = time.time()
                # train_result = trainer.train()
                epoch_duration = time.time() - epoch_start_time
                total_training_time += epoch_duration

                # # Calculate training speed
                # train_speed = len(self.formatted_dataset['train']) / epoch_duration
                # self.logger.info(f"Epoch {epoch + 1} training speed: {train_speed:.2f} steps/sec")

                # Evaluate the model
                train_accuracy = self.evaluate(self.model, self.formatted_dataset['train'])
                self.logger.info(f"Epoch {epoch + 1} training accuracy: {train_accuracy:.4f}")
                
                # Log loss
                train_loss = train_result.training_loss
                self.logger.info(f"Epoch {epoch + 1} training loss: {train_loss:.4f}")

                # Evaluate on the validation set
                val_accuracy = self.evaluate(self.model, self.formatted_dataset['test'])
                self.logger.info(f"Epoch {epoch + 1} validation accuracy: {val_accuracy:.4f}")

            self.logger.info("DPO training completed.")
            self.logger.info(f"Total training time: {total_training_time:.2f} seconds")

        except Exception as e:
            self.logger.error(f"Error during DPO training: {e}")
            raise e

        self.logger.info("Saving the fine-tuned DPO model...")
        try:
            trainer.model.save_pretrained(self.config['training']['dpo']['output_dir_clean'])
            self.logger.info(f"Model saved to {self.config['training']['dpo']['output_dir_clean']}")
        except Exception as e:
            self.logger.error(f"Error saving the model: {e}")
            raise e

        self.logger.info("Saving the tokenizer...")
        try:
            self.tokenizer.save_pretrained(self.config['training']['dpo']['output_dir_clean'])
            self.logger.info(f"Tokenizer saved to {self.config['training']['dpo']['output_dir_clean']}")
        except Exception as e:
            self.logger.error(f"Error saving the tokenizer: {e}")
            raise e
