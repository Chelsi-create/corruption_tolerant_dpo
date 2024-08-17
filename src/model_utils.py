import os
import torch
import logging
import yaml
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM, get_peft_model, PeftConfig
from datasets import load_from_disk
from trl import DPOTrainer

class ModelLoader:
    def __init__(self, sft_model_path, config, credentials):
        self.config = config
        self.credentials = credentials
        self.sft_model_path = sft_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = self.config.get('cache_dir', None)
        self.logger = logging.getLogger(__name__)
        self.training_adapter_name = "training_model"
        self.reference_adapter_name = "reference_model"

    def load_tokenizer(self):
        """Load and return the tokenizer."""
        self.logger.info("Loading tokenizer...")
        tokenizer = LlamaTokenizer.from_pretrained(
            self.config['model']['name'],
            cache_dir=self.cache_dir,
            use_auth_token=self.credentials.get('hugging_face', {}).get('token', True)
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
            use_auth_token=self.credentials.get('hugging_face', {}).get('token', True)
        )
        model.to(self.device)
        self.logger.info("Base model loaded and moved to device.")
        return model


    def load_sft_model(self):
        """Load the fine-tuned SFT model and its tokenizer."""
        sft_model_pth = self.sft_model_path
        self.logger.info(f"Attempting to load SFT model from {sft_model_pth}")
    
        if not os.path.exists(sft_model_pth):
            self.logger.error(f"Directory does not exist: {sft_model_pth}")
            return None, None
    
        try:
            # Load the PEFT configuration
            peft_config = PeftConfig.from_pretrained(sft_model_pth)
            peft_config.base_model_name_or_path = self.config['model']['name']
            
            # Determine the best available device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")
    
            # Load the base model and tokenizer
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                cache_dir=self.cache_dir,
                use_auth_token=self.credentials.get('hugging_face', {}).get('token', True)
            )

            
            tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, use_auth_token=self.credentials.get('hugging_face', {}).get('token', True), cache_dir=self.cache_dir)
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

            # Load the PEFT model
            model = PeftModel.from_pretrained(base_model, sft_model_pth)
    
            model.to(device)
            model.eval()
    
            self.logger.info("SFT model and tokenizer loaded successfully.")
            return model, tokenizer
    
        except Exception as e:
            self.logger.error(f"Failed to load SFT model or tokenizer: {str(e)}")
            return None, None

    
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
        if self.config['training']['sft'].get('output_dir'):
            # Load a fine-tuned model if the path is provided
            model, tokenizer = self.load_sft_model()
        else:
            # Otherwise, load the base model and prepare it with LoRA
            base_model = self.load_base_model()
            model = self.prepare_lora_model(base_model)

        self.logger.info("Model loaded successfully.")
        return model
