import torch
from transformers import TrainingArguments
from trl import DPOTrainer
import logging
import os
from src.model_utils import ModelLoader

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

        model_loader = ModelLoader(config, credentials)

        try:
            self.model = model_loader.load_model()
            self.reference_model = model_loader.load_sft_model(config['training']['sft']['output_dir'])
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise e

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.reference_model.to(device)
        self.logger.info(f"Models loaded and moved to {device}.")

    def train(self):
        self.logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=self.config['training']['dpo']['output_dir'],
            evaluation_strategy=self.config['training']['dpo']['evaluation_strategy'],
            learning_rate=self.config['training']['dpo']['learning_rate'],
            per_device_train_batch_size=self.config['training']['dpo']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['dpo']['per_device_eval_batch_size'],
            num_train_epochs=self.config['training']['dpo']['num_train_epochs'],
            weight_decay=self.config['training']['dpo']['weight_decay'],
            logging_dir=self.config['training']['dpo']['logging_dir'],
            fp16=torch.cuda.is_available()
        )

        self.logger.info("Initializing the DPO Trainer...")
        try:
            trainer = DPOTrainer(
                model=self.model,
                reference_model=self.reference_model,
                args=training_args,
                train_dataset=self.formatted_dataset['train'],
                eval_dataset=self.formatted_dataset['validation'],
                model_adapter_name="training_model",
                ref_adapter_name="reference_model"
            )
        except Exception as e:
            self.logger.error(f"Error initializing DPO Trainer: {e}")
            raise e

        self.logger.info("Starting DPO training...")
        try:
            trainer.train()
            self.logger.info("DPO training completed.")
        except Exception as e:
            self.logger.error(f"Error during DPO training: {e}")
            raise e

        self.logger.info("Saving the fine-tuned DPO model...")
        try:
            self.model.save_pretrained(self.config['training']['dpo']['output_dir'])
            self.logger.info(f"Model saved to {self.config['training']['dpo']['output_dir']}")
        except Exception as e:
            self.logger.error(f"Error saving the model: {e}")
            raise e

        if 'tokenizer' in self.formatted_dataset:
            try:
                tokenizer = self.formatted_dataset['tokenizer']
                tokenizer.save_pretrained(self.config['training']['dpo']['output_dir'])
                self.logger.info(f"Tokenizer saved to {self.config['training']['dpo']['output_dir']}")
            except Exception as e:
                self.logger.error(f"Error saving the tokenizer: {e}")
                raise e
