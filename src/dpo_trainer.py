import torch
from transformers import TrainingArguments
from trl import DPOTrainer
import logging
import os
from src.model_utils import ModelLoader
from peft import PeftConfig

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
                

    def train(self):
        self.logger.info("Setting up training arguments...")

        self.reference_model.eval()
        for param in self.model.parameters():
            param.requires_grad = True

        self.check_gradients(self.model)
        
        training_args = TrainingArguments(
            output_dir=self.config['training']['dpo']['output_dir'],
            evaluation_strategy=self.config['training']['dpo']['evaluation_strategy'],
            learning_rate=self.config['training']['dpo']['learning_rate'],
            per_device_train_batch_size=self.config['training']['dpo']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['dpo']['per_device_eval_batch_size'],
            num_train_epochs=self.config['training']['dpo']['num_train_epochs'],
            weight_decay=self.config['training']['dpo']['weight_decay'],
            logging_dir=self.config['training']['dpo']['logging_dir'],
            fp16=False,
            gradient_accumulation_steps=self.config['training']['dpo']['gradient_accumulation_steps'],
            offload_state_dict=True,
            offload_optimizer=True,
        )

        self.logger.info("Initializing the DPO Trainer...")
        try:
            trainer = DPOTrainer(
                model=self.model,
                ref_model=self.reference_model,
                args=training_args,
                train_dataset=self.formatted_dataset['train'],
                eval_dataset=self.formatted_dataset['validation'],
                tokenizer=self.tokenizer,
                beta=self.config['training']['dpo'].get('beta', 0.1),  # Add beta parameter
                max_prompt_length=self.config['training']['dpo'].get('max_prompt_length', 512),  # Add max_prompt_length
                max_length=self.config['training']['dpo'].get('max_length', 1024)
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
            trainer.model.save_pretrained(self.config['training']['dpo']['output_dir'])
            self.logger.info(f"Model saved to {self.config['training']['dpo']['output_dir']}")
        except Exception as e:
            self.logger.error(f"Error saving the model: {e}")
            raise e

        self.logger.info("Saving the tokenizer...")
        try:
            self.tokenizer.save_pretrained(self.config['training']['dpo']['output_dir'])
            self.logger.info(f"Tokenizer saved to {self.config['training']['dpo']['output_dir']}")
        except Exception as e:
            self.logger.error(f"Error saving the tokenizer: {e}")
            raise e
