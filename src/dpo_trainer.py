import torch
from transformers import TrainingArguments
from peft import AutoPeftModelForCausalLM
from trl import DPOTrainer

class DPOTrainerModule:
    def __init__(self, sft_model_path, formatted_dataset, config):
        self.config = config
        self.formatted_dataset = formatted_dataset

        self.model = AutoPeftModelForCausalLM.from_pretrained(sft_model_path)
        self.reference_model = AutoPeftModelForCausalLM.from_pretrained(sft_model_path)

        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.reference_model.to(device)

    def train(self):
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

        # Initialize the DPO Trainer
        trainer = DPOTrainer(
            model=self.model,
            reference_model=self.reference_model,
            args=training_args,
            train_dataset=self.formatted_dataset['train'],
            eval_dataset=self.formatted_dataset['validation']
        )

        # Train the model using Direct Policy Optimization (DPO)
        trainer.train()

        # Save the fine-tuned DPO model
        self.model.save_pretrained(self.config['training']['dpo']['output_dir'])
        print(f"Model saved to {self.config['training']['dpo']['output_dir']}")

        # Optionally, save the tokenizer if needed
        tokenizer = self.formatted_dataset['tokenizer']  # Assume tokenizer is part of the dataset
        tokenizer.save_pretrained(self.config['training']['dpo']['output_dir'])
        print(f"Tokenizer saved to {self.config['training']['dpo']['output_dir']}")