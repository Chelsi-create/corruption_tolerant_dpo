import torch
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        """Custom loss computation to ensure loss is returned correctly."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute the loss using CrossEntropy
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
        return loss

class SFTTrainer:
    def __init__(self, model, tokenized_dataset, config):
        self.config = config
        self.tokenized_dataset = tokenized_dataset

        # Prepare the LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Causal Language Modeling task
            r=8,                           # LoRA rank (adjustable hyperparameter)
            lora_alpha=16,                 # LoRA alpha (adjustable hyperparameter)
            lora_dropout=0.1,              # Dropout rate for LoRA (adjustable hyperparameter)
            target_modules=["q_proj", "v_proj"],  # Target modules to apply LoRA to
        )

        # Integrate LoRA with the model using PEFT
        self.model = get_peft_model(model, lora_config)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        # Define training arguments for SFT (Supervised Fine-Tuning)
        training_args = TrainingArguments(
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

        # Initialize the CustomTrainer for SFT
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation']
        )

        # Train the model using Supervised Fine-Tuning (SFT)
        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained(self.config['training']['sft']['output_dir'])
        print(f"Model saved to {self.config['training']['sft']['output_dir']}")

        # Optionally, save the tokenizer
        tokenizer = self.tokenized_dataset['tokenizer']  # Assume tokenizer is part of the dataset
        tokenizer.save_pretrained(self.config['training']['sft']['output_dir'])
        print(f"Tokenizer saved to {self.config['training']['sft']['output_dir']}")
