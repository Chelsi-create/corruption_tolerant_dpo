import torch
from transformers import TrainingArguments
from peft import AutoPeftModelForCausalLM
from trl import DPOTrainer
import logging
import os

class DPOTrainerModule:
    def __init__(self, sft_model_path, formatted_dataset, config):
        self.config = config
        self.formatted_dataset = formatted_dataset

        # Setup logging to a file
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

        # Fetch Hugging Face token from config if it exists
        hf_token = self.config.get('hugging_face', {}).get('token', None)
        if not hf_token:
            self.logger.warning("Hugging Face token not found in config. Ensure the model is accessible without authentication.")

        try:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                sft_model_path,
                use_auth_token=hf_token
            )
            self.reference_model = AutoPeftModelForCausalLM.from_pretrained(
                sft_model_path,
                use_auth_token=hf_token
            )
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise e

        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.reference_model.to(device)
        self.logger.info(f"Models loaded and moved to {device}.")

    def train(self):
        self.logger.info("Setting up training arguments...")
        # Define training arguments for DPO (Direct Policy Optimization)
        training_args = TrainingArguments(
            output_dir=self.config['training']['dpo']['output_dir'],
            evaluation_strategy=self.config['training']['dpo']['evaluation_strategy'],
            learning_rate=self.config['training']['dpo']['learning_rate'],
            per_device_train_batch_size=self.config['training']['dpo']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['dpo']['per_device_eval_batch_size'],
            num_train_epochs=self.config['training']['dpo']['num_train_epochs'],
            weight_decay=self.config['training']['dpo']['weight_decay'],
            logging_dir=self.config['training']['dpo']['logging_dir'],
            fp16=torch.cuda.is_available()  # Enable mixed precision if on GPU
        )

        self.logger.info("Initializing the DPO Trainer...")
        # Initialize the DPO Trainer
        try:
            trainer = DPOTrainer(
                model=self.model,
                reference_model=self.reference_model,
                args=training_args,
                train_dataset=self.formatted_dataset['train'],
                eval_dataset=self.formatted_dataset['validation']
            )
        except Exception as e:
            self.logger.error(f"Error initializing DPO Trainer: {e}")
            raise e

        self.logger.info("Starting DPO training...")
        # Train the model using Direct Policy Optimization (DPO)
        try:
            trainer.train()
            self.logger.info("DPO training completed.")
        except Exception as e:
            self.logger.error(f"Error during DPO training: {e}")
            raise e

        self.logger.info("Saving the fine-tuned DPO model...")
        # Save the fine-tuned DPO model
        try:
            self.model.save_pretrained(self.config['training']['dpo']['output_dir'])
            self.logger.info(f"Model saved to {self.config['training']['dpo']['output_dir']}")
        except Exception as e:
            self.logger.error(f"Error saving the model: {e}")
            raise e

        # Optionally, save the tokenizer if needed
        if 'tokenizer' in self.formatted_dataset:
            try:
                tokenizer = self.formatted_dataset['tokenizer']  # Assume tokenizer is part of the dataset
                tokenizer.save_pretrained(self.config['training']['dpo']['output_dir'])
                self.logger.info(f"Tokenizer saved to {self.config['training']['dpo']['output_dir']}")
            except Exception as e:
                self.logger.error(f"Error saving the tokenizer: {e}")
                raise e
