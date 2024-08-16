import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = self.config.get('cache_dir', None)  # Optional cache directory

    def load_tokenizer(self):
        """Load and return the tokenizer."""
        tokenizer = LlamaTokenizer.from_pretrained(
            self.config['model']['name'],
            cache_dir=self.cache_dir,  # Use custom cache directory if provided
            use_auth_token=True  # Use authentication token for gated models
        )
        return tokenizer

    def load_base_model(self):
        """Load the base LLaMA 2 model."""
        model = LlamaForCausalLM.from_pretrained(
            self.config['model']['name'],
            cache_dir=self.cache_dir,  # Use custom cache directory if provided
            use_auth_token=True  # Use authentication token for gated models
        )
        model.to(self.device)
        return model

    def prepare_lora_model(self, model):
        """Prepare and return the LLaMA 2 model with LoRA configuration applied."""
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
        return lora_model

    def load_sft_model(self, sft_model_path):
        """Load the fine-tuned SFT model using PEFT's AutoPeftModelForCausalLM."""
        model = AutoPeftModelForCausalLM.from_pretrained(
            sft_model_path,
            cache_dir=self.cache_dir,  # Use custom cache directory if provided
            use_auth_token=True  # Use authentication token for gated models
        )
        model.to(self.device)
        return model

    def load_model(self):
        """
        Load the model based on the configuration.
        If a fine-tuned model path is provided, load that model.
        Otherwise, prepare a LoRA model from the base model.
        """
        if self.config['model'].get('sft_model_path'):
            # Load a fine-tuned model if the path is provided
            model = self.load_sft_model(self.config['model']['sft_model_path'])
        else:
            # Otherwise, load the base model and prepare it with LoRA
            base_model = self.load_base_model()
            model = self.prepare_lora_model(base_model)
        
        return model
