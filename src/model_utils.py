import os
import torch
import logging
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model, AutoPeftModelForCausalLM

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = self.config.get('cache_dir', None)
        self.logger = logging.getLogger(__name__)
        self.sft_model_path = self.config['training']['sft']['output_dir']
        self.training_adapter_name = "training_model"
        self.reference_adapter_name = "reference_model"

    def load_tokenizer(self):
        """Load and return the tokenizer."""
        self.logger.info("Loading tokenizer...")
        tokenizer = LlamaTokenizer.from_pretrained(
            self.config['model']['name'],
            cache_dir=self.cache_dir,
            use_auth_token=self.config.get('hugging_face', {}).get('token', True)
        )
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
        self.logger.info("Tokenizer loaded successfully.")
        return tokenizer

    def load_base_model(self):
        """Load the base LLaMA 2 model."""
        self.logger.info("Loading base model...")
        model = LlamaForCausalLM.from_pretrained(
            self.config['model']['name'],
            cache_dir=self.cache_dir,
            use_auth_token=self.config.get('hugging_face', {}).get('token', True)
        )
        model.to(self.device)
        self.logger.info("Base model loaded and moved to device.")
        return model

    def load_sft_model(self, sft_model_pth):
        """Load the fine-tuned SFT model using PEFT's AutoPeftModelForCausalLM."""
        self.logger.info(f"Loading SFT model from {sft_model_pth}...")
        full_model_pth = os.path.abspath(sft_model_pth)

        if not os.path.exists(full_model_pth):
            self.logger.error(f"Directory does not exist: {full_model_pth}")
            raise FileNotFoundError(f"SFT model directory not found: {full_model_pth}")

        self.logger.info(f"Directory exists. Contents: {os.listdir(full_model_pth)}")

        # Load the SFT model with the training adapter
        model = AutoPeftModelForCausalLM.from_pretrained(
            full_model_pth,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Attach the PEFT adapters for training and reference
        model = PeftModel.from_pretrained(model, full_model_pth, is_trainable=True, adapter_name=self.training_adapter_name)
        model.load_adapter(full_model_pth, adapter_name=self.reference_adapter_name)

        self.logger.info(f"SFT model loaded successfully. Model type: {type(model).__name__}")
        return model

    def prepare_lora_model(self, model):
        """Prepare and return the LLaMA 2 model with LoRA configuration applied."""
        self.logger.info("Preparing LoRA model...")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )
        lora_model = get_peft_model(model, lora_config)
        lora_model.to(self.device)
        self.logger.info("LoRA model prepared successfully.")
        return lora_model

    def load_model(self):
        """Load the model based on the configuration."""
        self.logger.info("Loading model based on configuration...")
        if self.config['model'].get('sft_model_path'):
            # Load a fine-tuned model if the path is provided
            model = self.load_sft_model(self.config['model']['sft_model_path'])
        else:
            # Otherwise, load the base model and prepare it with LoRA
            base_model = self.load_base_model()
            model = self.prepare_lora_model(base_model)

        self.logger.info("Model loaded successfully.")
        return model


def train_dpo(config):
    # Initialize the model loader
    model_loader = ModelLoader(config)

    # Load the tokenizer and model
    tokenizer = model_loader.load_tokenizer()
    model = model_loader.load_model()

    # Load the dataset
    dataset = load_from_disk(config['dataset']['load_dir'])

    # Define the training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        remove_unused_columns=False,
        num_train_epochs=config['training']['dpo']['epochs'],
        output_dir=config['training']['dpo']['save_dir'],
        save_steps=2000,
        logging_first_step=True,
        logging_steps=50,
        learning_rate=config['training']['dpo']['learning_rate'],
        optim="rmsprop",
        bf16=True,
    )

    # Initialize the DPO Trainer
    dpo_trainer = DPOTrainer(
        model,
        model_adapter_name="training_model",
        ref_adapter_name="reference_model",
        args=training_args,
        beta=config['training']['dpo']['beta'],
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_target_length=1024,
        max_prompt_length=1024,
    )

    # Start training
    dpo_trainer.train()

    # Save the trained model
    dpo_trainer.model.save_pretrained(config['training']['dpo']['save_dir'], from_pt=True)


# Example usage
if __name__ == "__main__":
    config_path = "path/to/config.yaml"  # Replace with your config path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_dpo(config)
