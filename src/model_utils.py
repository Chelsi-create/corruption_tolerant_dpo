import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM
import logging
import os

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = self.config.get('cache_dir', None)  # Optional cache directory
        self.logger = logging.getLogger(__name__)
        self.sft_model_path = self.config['training']['sft']['output_dir']

    def load_tokenizer(self):
        """Load and return the tokenizer."""
        self.logger.info("Loading tokenizer...")
        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                self.config['model']['name'],
                cache_dir=self.cache_dir,  # Use custom cache directory if provided
                use_auth_token=self.config['hugging_face']['token'] if 'hugging_face' in self.config and 'token' in self.config['hugging_face'] else True  # Use authentication token for gated models
            )
            self.logger.info("Tokenizer loaded successfully.")
            return tokenizer
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise

    def load_base_model(self):
        """Load the base LLaMA 2 model."""
        self.logger.info("Loading base model...")
        try:
            model = LlamaForCausalLM.from_pretrained(
                self.config['model']['name'],
                cache_dir=self.cache_dir,  # Use custom cache directory if provided
                use_auth_token=self.config['hugging_face']['token'] if 'hugging_face' in self.config and 'token' in self.config['hugging_face'] else True  # Use authentication token for gated models
            )
            model.to(self.device)
            self.logger.info("Base model loaded and moved to device.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            raise

    def prepare_lora_model(self, model):
        """Prepare and return the LLaMA 2 model with LoRA configuration applied."""
        self.logger.info("Preparing LoRA model...")
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,  # Causal Language Modeling task
                r=8,                           # LoRA rank (adjustable hyperparameter)
                lora_alpha=16,                 # LoRA alpha (adjustable hyperparameter)
                lora_dropout=0.1,              # Dropout rate for LoRA (adjustable hyperparameter)
                target_modules=["q_proj", "v_proj"],  # Target modules to apply LoRA to
            )

            # Integrate LoRA with the model using PEFT
            lora_model = get_peft_model(model, lora_config)
            lora_model.to(self.device)
            self.logger.info("LoRA model prepared successfully.")
            return lora_model
        except Exception as e:
            self.logger.error(f"Failed to prepare LoRA model: {e}")
            raise
    
    def load_sft_model(self, sft_model_pth):
        """Load the fine-tuned SFT model using PEFT's AutoPeftModelForCausalLM."""
        self.logger.info(f"Loading SFT model from {sft_model_pth}...")
        try:
            if os.path.isdir(sft_model_pth):
                # It's a local directory, so no need for a token
                model = AutoPeftModelForCausalLM.from_pretrained(
                    sft_model_pth,
                    cache_dir=self.cache_dir  # Use custom cache directory if provided
                )
            else:
                # It's a model name or path from Hugging Face Hub, require a token
                model = AutoPeftModelForCausalLM.from_pretrained(
                    sft_model_pth,
                    cache_dir=self.cache_dir,  # Use custom cache directory if provided
                    use_auth_token=self.config['hugging_face']['token'] if 'hugging_face' in self.config and 'token' in self.config['hugging_face'] else True  # Use authentication token for gated models
                )
            model.to(self.device)
            self.logger.info("SFT model loaded and moved to device.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load SFT model: {e}")
            raise


    def load_model(self):
        """
        Load the model based on the configuration.
        If a fine-tuned model path is provided, load that model.
        Otherwise, prepare a LoRA model from the base model.
        """
        self.logger.info("Loading model based on configuration...")
        try:
            if self.sft_model_path:
                # Load a fine-tuned model if the path is provided
                model = self.load_sft_model(self.sft_model_path)
            else:
                # Otherwise, load the base model and prepare it with LoRA
                base_model = self.load_base_model()
                model = self.prepare_lora_model(base_model)
            
            self.logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
